import os
import json
import argparse
import re
import random
from dotenv import load_dotenv
from otto import OttoAgent

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Shared helpers for grid parsing/normalization
def strip_code_fences(text: str) -> str:
    text = text.strip()
    if "```" in text:
        last = text.rfind("```")
        second = text.rfind("```", 0, last)
        if second != -1 and last != -1 and second < last:
            return text[second + 3:last].strip()
    return text


def parse_grid_from_text(text: str):
    text = text.strip()
    # Try JSON first
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        pass

    # Fallback: regex parse integers per line
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        numbers = [int(tok) for tok in re.findall(r"-?\d+", line)]
        if numbers:
            rows.append(numbers)
    return rows


def is_2d_int_grid(obj) -> bool:
    return (
        isinstance(obj, list)
        and len(obj) > 0
        and all(isinstance(row, list) and len(row) > 0 for row in obj)
        and all(isinstance(val, (int, float)) for row in obj for val in row)
    )


def normalize_grid_structure(obj):
    # Accept either 2D grid or [grid]
    if is_2d_int_grid(obj):
        return [[int(v) for v in row] for row in obj]
    if isinstance(obj, list) and len(obj) > 0 and is_2d_int_grid(obj[0]):
        return [[int(v) for v in row] for row in obj[0]]
    return obj


def is_rectangular(grid) -> bool:
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    if not all(isinstance(row, list) and len(row) > 0 for row in grid):
        return False
    row_len = len(grid[0])
    return all(len(row) == row_len for row in grid)


def parse_and_normalize_grid(grid_or_text: str, require_rectangular: bool = False):
    stripped = strip_code_fences(grid_or_text) if isinstance(grid_or_text, str) else grid_or_text
    parsed = parse_grid_from_text(stripped) if isinstance(stripped, str) else stripped
    normalized = normalize_grid_structure(parsed)
    if not is_2d_int_grid(normalized):
        raise ValueError("Not a 2D integer grid structure.")
    if require_rectangular and not is_rectangular(normalized):
        raise ValueError("Grid is not rectangular.")
    return normalized


def read_task(task_id):
    with open("public_dataset/arc-agi_training_challenges.json", "r") as f:
        data = json.load(f)
    task_data = data[task_id]

    # Format the task as a multi-line string
    result = f"# Task {task_id}\n\n"

    result += " ".join([
        "## Description\n\n",
        "This is an ARC AGI challenge.",
        "Your goal is to identify the transformation logic that is shared across all training examples below.",
        "Then, apply this logic to the challenge input to produce the correct output.\n\n",
    ])

    result += f"## Training Examples\n\n"

    # Add training examples
    for i, example in enumerate(task_data["train"], 1):
        result += f"### Example {i}\n\n"
        result += "Input:\n```\n"
        for row in example["input"]:
            result += " ".join(map(str, row)) + "\n"
        result += "\nOutput:\n```\n"
        for row in example["output"]:
            result += " ".join(map(str, row)) + "\n"
        result += "```\n\n"

    # Add test example if it exists
    if "test" in task_data and task_data["test"]:
        result += "## Challenge\n\n"
        test_example = task_data["test"][0]  # Usually only one test example
        result += "Input:\n```\n"
        for row in test_example["input"]:
            result += " ".join(map(str, row)) + "\n"
        result += "```\n\n"

    return result


# Define the read_task tool
read_task_tool = {
    "type": "function",
    "function": {
        "name": "read_task",
        "description": "Read and format an ARC AGI task from the training dataset by task ID",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "The task ID/key from the ARC training challenges"}
            },
            "required": ["task_id"]
        },
    },
}

def read_task_handler(tool_call: dict) -> dict:
    """Handler for the read_task tool"""
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    task_id = str(args.get("task_id", "")).strip()

    try:
        result = read_task(task_id)
        response = {"ok": True, "task_description": result}
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "read_task",
        "content": json.dumps(response),
    }

# Define the validate_answer tool
validate_answer_tool = {
    "type": "function",
    "function": {
        "name": "validate_answer",
        "description": "Validate that the provided answer text represents a rectangular integer grid.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer_text": {"type": "string", "description": "The model's proposed final answer grid (may include code fences)."}
            },
            "required": ["answer_text"]
        },
    },
}

def validate_answer_handler(tool_call: dict) -> dict:
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    answer_text = str(args.get("answer_text", ""))

    def strip_code_fences(text: str) -> str:
        text = text.strip()
        if "```" in text:
            last = text.rfind("```")
            second = text.rfind("```", 0, last)
            if second != -1 and last != -1 and second < last:
                return text[second + 3:last].strip()
        return text

    def parse_grid_from_text(text: str):
        text = text.strip()
        # Try JSON first
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            pass

        # Fallback: regex parse integers per line
        rows = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            numbers = [int(tok) for tok in re.findall(r"-?\d+", line)]
            if numbers:
                rows.append(numbers)
        return rows

    def is_2d_int_grid(obj) -> bool:
        return (
            isinstance(obj, list)
            and len(obj) > 0
            and all(isinstance(row, list) and len(row) > 0 for row in obj)
            and all(isinstance(val, (int, float)) for row in obj for val in row)
        )

    def normalize_grid(obj):
        # Accept either 2D grid or [grid]
        if is_2d_int_grid(obj):
            return [[int(v) for v in row] for row in obj]
        if isinstance(obj, list) and len(obj) > 0 and is_2d_int_grid(obj[0]):
            return [[int(v) for v in row] for row in obj[0]]
        return obj

    def is_rectangular(grid) -> bool:
        if not isinstance(grid, list) or len(grid) == 0:
            return False
        if not all(isinstance(row, list) and len(row) > 0 for row in grid):
            return False
        row_len = len(grid[0])
        return all(len(row) == row_len for row in grid)

    try:
        stripped = strip_code_fences(answer_text)
        parsed = parse_grid_from_text(stripped)
        normalized = normalize_grid(parsed)

        if not is_2d_int_grid(normalized):
            response = {"ok": False, "error": "Answer is not a 2D integer grid."}
        elif not is_rectangular(normalized):
            response = {"ok": False, "error": "Grid is not rectangular (rows have differing lengths)."}
        else:
            rows = len(normalized)
            cols = len(normalized[0])
            response = {
                "ok": True,
                "is_rectangular": True,
                "rows": rows,
                "cols": cols,
                "grid": normalized,
            }
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "validate_answer",
        "content": json.dumps(response),
    }

# normalize_answer tool
normalize_answer_tool = {
    "type": "function",
    "function": {
        "name": "normalize_answer",
        "description": "Normalize answer text into a rectangular 2D integer grid and return it with shape.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer_text": {"type": "string", "description": "Answer grid text or JSON (may include code fences)"}
            },
            "required": ["answer_text"]
        },
    },
}

def normalize_answer_handler(tool_call: dict) -> dict:
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    answer_text = str(args.get("answer_text", ""))

    try:
        grid = parse_and_normalize_grid(answer_text, require_rectangular=True)
        response = {
            "ok": True,
            "rows": len(grid),
            "cols": len(grid[0]) if grid else 0,
            "grid": grid,
        }
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "normalize_answer",
        "content": json.dumps(response),
    }

# compare_grids tool
compare_grids_tool = {
    "type": "function",
    "function": {
        "name": "compare_grids",
        "description": "Compare two grids (text or JSON) for equality; return shape and mismatch stats.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "Grid A as text or JSON (may include code fences)"},
                "b": {"type": "string", "description": "Grid B as text or JSON (may include code fences)"}
            },
            "required": ["a", "b"]
        },
    },
}

def compare_grids_handler(tool_call: dict) -> dict:
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    a = args.get("a", "")
    b = args.get("b", "")
    try:
        grid_a = parse_and_normalize_grid(a, require_rectangular=True)
        grid_b = parse_and_normalize_grid(b, require_rectangular=True)

        same_shape = (len(grid_a) == len(grid_b)) and all(len(grid_a[r]) == len(grid_b[r]) for r in range(len(grid_a)))
        if not same_shape:
            response = {
                "ok": True,
                "equal": False,
                "reason": "Different shapes",
                "a_shape": [len(grid_a), len(grid_a[0]) if grid_a else 0],
                "b_shape": [len(grid_b), len(grid_b[0]) if grid_b else 0],
            }
        else:
            mismatches = 0
            first_mismatch = None
            for r in range(len(grid_a)):
                for c in range(len(grid_a[0])):
                    if int(grid_a[r][c]) != int(grid_b[r][c]):
                        mismatches += 1
                        if first_mismatch is None:
                            first_mismatch = {"row": r, "col": c, "a": int(grid_a[r][c]), "b": int(grid_b[r][c])}
            response = {
                "ok": True,
                "equal": mismatches == 0,
                "mismatch_count": mismatches,
                "first_mismatch": first_mismatch,
                "shape": [len(grid_a), len(grid_a[0]) if grid_a else 0],
            }
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "compare_grids",
        "content": json.dumps(response),
    }

# grid_shape tool
grid_shape_tool = {
    "type": "function",
    "function": {
        "name": "grid_shape",
        "description": "Return the (rows, cols) of a grid (text or JSON).",
        "parameters": {
            "type": "object",
            "properties": {
                "grid": {"type": "string", "description": "Grid as text or JSON (may include code fences)"}
            },
            "required": ["grid"]
        },
    },
}

def grid_shape_handler(tool_call: dict) -> dict:
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    grid_text = args.get("grid", "")
    try:
        grid = parse_and_normalize_grid(grid_text, require_rectangular=True)
        response = {"ok": True, "rows": len(grid), "cols": len(grid[0]) if grid else 0}
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "grid_shape",
        "content": json.dumps(response),
    }

# color_histogram tool
color_histogram_tool = {
    "type": "function",
    "function": {
        "name": "color_histogram",
        "description": "Return counts of each color present in the grid.",
        "parameters": {
            "type": "object",
            "properties": {
                "grid": {"type": "string", "description": "Grid as text or JSON (may include code fences)"}
            },
            "required": ["grid"]
        },
    },
}

def color_histogram_handler(tool_call: dict) -> dict:
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    grid_text = args.get("grid", "")
    try:
        grid = parse_and_normalize_grid(grid_text)
        hist = {}
        for row in grid:
            for v in row:
                v = int(v)
                hist[str(v)] = hist.get(str(v), 0) + 1
        response = {"ok": True, "histogram": hist}
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "color_histogram",
        "content": json.dumps(response),
    }

# bounding_boxes tool
bounding_boxes_tool = {
    "type": "function",
    "function": {
        "name": "bounding_boxes",
        "description": "Compute per-color bounding boxes, or only for a given color.",
        "parameters": {
            "type": "object",
            "properties": {
                "grid": {"type": "string", "description": "Grid as text or JSON (may include code fences)"},
                "color": {"type": "integer", "description": "Optional color to filter to"}
            },
            "required": ["grid"]
        },
    },
}

def bounding_boxes_handler(tool_call: dict) -> dict:
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    grid_text = args.get("grid", "")
    color = args.get("color", None)
    try:
        grid = parse_and_normalize_grid(grid_text)
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        boxes = {}
        for r in range(rows):
            for c in range(cols):
                v = int(grid[r][c])
                if color is not None and v != int(color):
                    continue
                key = str(v)
                if key not in boxes:
                    boxes[key] = {"min_row": r, "min_col": c, "max_row": r, "max_col": c}
                else:
                    b = boxes[key]
                    b["min_row"] = min(b["min_row"], r)
                    b["min_col"] = min(b["min_col"], c)
                    b["max_row"] = max(b["max_row"], r)
                    b["max_col"] = max(b["max_col"], c)
        response = {"ok": True, "boxes": boxes}
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "bounding_boxes",
        "content": json.dumps(response),
    }

# find_symmtery tool (intentionally matching requested name)
find_symmtery_tool = {
    "type": "function",
    "function": {
        "name": "find_symmtery",
        "description": "Detect horizontal, vertical, and 180° rotational symmetry on the grid.",
        "parameters": {
            "type": "object",
            "properties": {
                "grid": {"type": "string", "description": "Grid as text or JSON (may include code fences)"}
            },
            "required": ["grid"]
        },
    },
}

def find_symmtery_handler(tool_call: dict) -> dict:
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    grid_text = args.get("grid", "")
    try:
        g = parse_and_normalize_grid(grid_text, require_rectangular=True)
        rows = len(g)
        cols = len(g[0]) if rows else 0

        def eq(a, b):
            return int(a) == int(b)

        # Horizontal symmetry (mirror around vertical axis)
        horiz = True
        for r in range(rows):
            for c in range(cols // 2):
                if not eq(g[r][c], g[r][cols - 1 - c]):
                    horiz = False
                    break
            if not horiz:
                break

        # Vertical symmetry (mirror around horizontal axis)
        vert = True
        for r in range(rows // 2):
            for c in range(cols):
                if not eq(g[r][c], g[rows - 1 - r][c]):
                    vert = False
                    break
            if not vert:
                break

        # 180-degree rotational symmetry
        rot180 = True
        for r in range(rows):
            for c in range(cols):
                if not eq(g[r][c], g[rows - 1 - r][cols - 1 - c]):
                    rot180 = False
                    break
            if not rot180:
                break

        response = {"ok": True, "horizontal": horiz, "vertical": vert, "rotational_180": rot180}
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "find_symmtery",
        "content": json.dumps(response),
    }

# crop tool
crop_tool = {
    "type": "function",
    "function": {
        "name": "crop",
        "description": "Crop a rectangular region from the grid using inclusive rows/cols.",
        "parameters": {
            "type": "object",
            "properties": {
                "grid": {"type": "string", "description": "Grid as text or JSON (may include code fences)"},
                "min_row": {"type": "integer"},
                "min_col": {"type": "integer"},
                "max_row": {"type": "integer"},
                "max_col": {"type": "integer"}
            },
            "required": ["grid", "min_row", "min_col", "max_row", "max_col"]
        },
    },
}

def crop_handler(tool_call: dict) -> dict:
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    grid_text = args.get("grid", "")
    min_row = int(args.get("min_row", 0))
    min_col = int(args.get("min_col", 0))
    max_row = int(args.get("max_row", -1))
    max_col = int(args.get("max_col", -1))
    try:
        g = parse_and_normalize_grid(grid_text, require_rectangular=True)
        rows = len(g)
        cols = len(g[0]) if rows else 0
        if not (0 <= min_row <= max_row < rows and 0 <= min_col <= max_col < cols):
            raise ValueError("Crop bounds out of range.")
        cropped = [row[min_col:max_col + 1] for row in g[min_row:max_row + 1]]
        response = {"ok": True, "rows": len(cropped), "cols": len(cropped[0]) if cropped else 0, "grid": cropped}
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "crop",
        "content": json.dumps(response),
    }

# validate_dimensions tool
validate_dimensions_tool = {
    "type": "function",
    "function": {
        "name": "validate_dimensions",
        "description": "Validate that a proposed grid matches expected (rows, cols) dimensions.",
        "parameters": {
            "type": "object",
            "properties": {
                "grid": {"type": "string", "description": "Proposed grid as text or JSON (may include code fences)"},
                "rows": {"type": "integer", "description": "Expected number of rows"},
                "cols": {"type": "integer", "description": "Expected number of columns"}
            },
            "required": ["grid", "rows", "cols"]
        },
    },
}

def validate_dimensions_handler(tool_call: dict) -> dict:
    fn = tool_call.get("function", {})
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw)
    except Exception:
        args = {}
    grid_text = args.get("grid", "")
    expected_rows = int(args.get("rows", -1))
    expected_cols = int(args.get("cols", -1))
    try:
        g = parse_and_normalize_grid(grid_text, require_rectangular=True)
        rows = len(g)
        cols = len(g[0]) if rows else 0
        ok = (rows == expected_rows) and (cols == expected_cols)
        response = {"ok": ok, "rows": rows, "cols": cols}
        if not ok:
            response["error"] = "Dimensions do not match expected values."
    except Exception as e:
        response = {"ok": False, "error": str(e)}

    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": "validate_dimensions",
        "content": json.dumps(response),
    }

client = OttoAgent(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    extra_tools=[
        read_task_tool,
        validate_answer_tool,
        normalize_answer_tool,
        compare_grids_tool,
        grid_shape_tool,
        color_histogram_tool,
        bounding_boxes_tool,
        find_symmtery_tool,
        crop_tool,
        validate_dimensions_tool,
    ],
    extra_tool_handlers={
        "read_task": read_task_handler,
        "validate_answer": validate_answer_handler,
        "normalize_answer": normalize_answer_handler,
        "compare_grids": compare_grids_handler,
        "grid_shape": grid_shape_handler,
        "color_histogram": color_histogram_handler,
        "bounding_boxes": bounding_boxes_handler,
        "find_symmtery": find_symmtery_handler,
        "crop": crop_handler,
        "validate_dimensions": validate_dimensions_handler,
    },
)

def extract_solution(response):
    # Robustly extract the last code block that parses as a rectangular 2D grid
    # Try all fenced blocks (with optional language tag), keep the last valid one
    code_blocks = []
    try:
        # Match ```<lang>?\n...\n``` capturing inner content
        for m in re.finditer(r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)\s*```", response):
            code_blocks.append(m.group(1))
    except Exception:
        pass

    chosen = None
    for block in code_blocks:
        try:
            grid = parse_and_normalize_grid(block, require_rectangular=True)
            if grid and isinstance(grid, list) and isinstance(grid[0], list):
                chosen = block  # keep last valid
        except Exception:
            continue

    if chosen is not None:
        return chosen

    # Fallback: original heuristic using the last pair of backticks
    last_backtick = response.rfind("```")
    second_to_last_backtick = response.rfind("```", 0, last_backtick)
    if last_backtick != -1 and second_to_last_backtick != -1 and second_to_last_backtick < last_backtick:
        return response[second_to_last_backtick + 3:last_backtick]
    return response


def compare_solution(task_id, given_solution):
    # Read expected grid and compare using shared normalization utilities
    with open(f"DO_NOT_OPEN/arc-agi_training_solutions.json", "r") as f:
        data = json.load(f)
    expected_obj = data[task_id]

    try:
        given_grid = parse_and_normalize_grid(given_solution, require_rectangular=True)
    except Exception:
        # Try stripping fences then parse again
        given_grid = parse_and_normalize_grid(strip_code_fences(given_solution), require_rectangular=True)

    expected_grid = parse_and_normalize_grid(expected_obj, require_rectangular=True)

    # Compare dimensions first
    if len(given_grid) != len(expected_grid):
        return False
    for r in range(len(expected_grid)):
        if len(given_grid[r]) != len(expected_grid[r]):
            return False

    # Compare cell-by-cell
    for r in range(len(expected_grid)):
        for c in range(len(expected_grid[r])):
            if int(given_grid[r][c]) != int(expected_grid[r][c]):
                return False

    return True

def retrieve_solution(task_id):
    with open(f"DO_NOT_OPEN/arc-agi_training_solutions.json", "r") as f:
        data = json.load(f)
    # Normalize to a 2D grid and pretty-print
    grid = parse_and_normalize_grid(data[task_id], require_rectangular=True)
    for row in grid:
        print(" ".join(map(str, row)))
    return grid


def solve_task(task_id):
    # Use Otto to read and solve the task
    print("=== Solving task", task_id, "===")
    print("\n\n=== Step 1: Synthesizing transformation logic ===")
    prompt = (
        f"We are solving an ARC AGI challenge. The task id is {task_id}. These challenges involve a grid of integers, representing different colored cells. "
        "You are given a set of training examples, each containing an input grid and an output grid. "
        "All training examples share the same tranformation logic. Most often, the transformation rule can be described in one or two short sentences. "
        "For example, a transformation rule might be 'use the input grid as a stamp, and apply the stamp inverted in locations correlating to the input grid colors.' "
        "Another example might be 'copy the input grid as the output grid, then {some additional logic like flood flill, or extend vectors, etc}'. "
        "--- "
        "To begin, use the `read_task` tool to view the training examples and challenge input. "
        "For now, we are not going to solve the challenge. Right now, we are trying to synthenize  a transformation logic that can be applied unformily across all the training examples."
        "Use the other task-specifictools available to help you understand the transformation logic and validate your understanding. Do not attempt to read files or do any read-based operations." 
        "Once you are ready, simply state the transfomration logic concisely. Then stop. "
    )
    response = client.prompt(prompt, verbose=True)
    print("\n\n=== Step 2: Understanding the challenge input ===")
    prompt = (
        "Now that we know the transformation logic, please re-read the task and challenge input with the `read_task` tool."
        "Then, reason through the challenge input. Understand the input grid, and what you expect the output grid to generally look like."
        "Describe what the output grid should look like, without actually outputting the grid itself. Then stop."
    )
    response = client.prompt(prompt, verbose=True)
    print("\n\n=== Step 3: Determine the output grid dimensions ===")
    prompt = (
        "Now that we know the transformation logic, please re-read the task and challenge input with the `read_task` tool."
        "We have already determined what we expect the output grid to look like. The input grid is in the above task data."
        "Now, determine the dimensions of the output grid. Do not actually output the grid itself, just the dimensions. Then stop."
    )
    response = client.prompt(prompt, verbose=True)
    print("\n\n=== Step 4: Providing the final solution ===")
    prompt = (
        "Now that we know the transformation logic of the examples, what the challenge output should look like, and the dimensions of the output grid, please provide the final solution output grid in a code block (```)."
        "To do this, use the `validate_dimensions` tool to validate the dimensions of your output grid. Ensure that the result of that tool passes before your provide the final solution."
        "At the end of your final response, ensure you provide the **complete** output grid in a code block enclosed by ```. Please don't truncate the output, as it will be used to check if your answer is correct."
    )
    response = client.prompt(prompt, verbose=True)
    print("\n\n=== Step 5: Checking the answer ===")
    # Check the answer
    given_solution = extract_solution(response["final_text"])
    if compare_solution(task_id, given_solution):
        print("\nCorrect!")
        return "correct"
    else:
        print("\nIncorrect!")
        print(f"The correct solution is: \n{retrieve_solution(task_id)}")
        return "incorrect"


def run_benchmark(num_tasks: int):
    with open("public_dataset/arc-agi_training_challenges.json", "r") as f:
        challenges = json.load(f)
    all_task_ids = list(challenges.keys())
    sample_size = min(max(int(num_tasks), 0), len(all_task_ids))
    sampled_ids = random.sample(all_task_ids, sample_size)

    correct_ids = []
    incorrect_ids = []
    error_ids = []

    for task_id in sampled_ids:
        print(
            f"Starting task {task_id} | correct: {len(correct_ids)} incorrect: {len(incorrect_ids)} error: {len(error_ids)}"
        )
        try:
            status = solve_task(task_id)
            if status == "correct":
                correct_ids.append(task_id)
            else:
                incorrect_ids.append(task_id)
        except Exception as e:
            print(f"Error solving task {task_id}: {e}")
            error_ids.append(task_id)

    results = {
        "summary": {
            "attempted": len(sampled_ids),
            "correct": len(correct_ids),
            "incorrect": len(incorrect_ids),
            "error": len(error_ids),
        },
        "correct": correct_ids,
        "incorrect": incorrect_ids,
        "error": error_ids,
    }

    with open("benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Benchmark results saved to benchmark.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC Agent - solve a specific task by id/key")
    parser.add_argument("--task", required=False, help="Task id/key to solve (matches key in training challenges)")
    parser.add_argument("--benchmark", type=int, required=False, help="Run a benchmark over N random tasks")
    args = parser.parse_args()
    if args.benchmark and args.benchmark > 0:
        run_benchmark(args.benchmark)
    else:
        task_id = args.task
        if not task_id:
            with open("public_dataset/arc-agi_training_challenges.json", "r") as f:
                challenges = json.load(f)
            task_id = random.choice(list(challenges.keys()))
            print(f"No task provided. Randomly selected task '{task_id}'.")
        solve_task(task_id)