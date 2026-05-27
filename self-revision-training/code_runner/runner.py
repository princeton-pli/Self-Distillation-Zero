import re
import sys
import io
import contextlib
import signal
import tempfile
import subprocess
import os
import json
from typing import List, Dict, Any, Tuple, Optional

# Language configuration
LANGUAGE_CONFIG = {
    "python": {
        "extensions": [".py"],
        "compile_cmd": None,
        "run_cmd": [sys.executable],
        "aliases": ["python", "py", "python3"]
    },
    "cpp": {
        "extensions": [".cpp"],
        "compile_cmd": ["g++", "-O3", "-o", "{exe}", "{src}"],
        "run_cmd": ["{exe}"],
        "aliases": ["cpp", "c++", "cc", "cxx"]
    },
    "c": {
        "extensions": [".c"],
        "compile_cmd": ["gcc", "-O3", "-o", "{exe}", "{src}"],
        "run_cmd": ["{exe}"],
        "aliases": ["c"]
    },
    "java": {
        "extensions": [".java"],
        "compile_cmd": ["javac", "{src}"],
        "run_cmd": ["java", "-cp", "{dir}", "{main_class}"],
        "aliases": ["java"]
    },
    "go": {
        "extensions": [".go"],
        "compile_cmd": ["go", "build", "-o", "{exe}", "{src}"],
        "run_cmd": ["{exe}"],
        "aliases": ["go", "golang"]
    },
    "rust": {
        "extensions": [".rs"],
        "compile_cmd": ["rustc", "-O", "-o", "{exe}", "{src}"],
        "run_cmd": ["{exe}"],
        "aliases": ["rust", "rs"]
    }
}

def extract_code_with_lang(response: str) -> Tuple[str, str]:
    """
    Extract code and language from the model's response.
    Returns (code, language). Language defaults to 'python' if not specified.
    """
    # Remove <think>...</think> blocks
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Look for code blocks with language specifier
    # e.g. ```cpp ... ```
    # We iterate over all matches to find the last one or the most relevant one?
    # Usually the last one is the final answer, but sometimes there are multiple blocks.
    # Let's take the last block that matches a known language, or just the last block.
    
    code_matches = list(re.finditer(r'```(\w+)\s*(.*?)```', response, re.DOTALL))
    if code_matches:
        # Check if any match is a known language
        for match in reversed(code_matches):
            lang = match.group(1).lower()
            code = match.group(2).strip()
            
            # Normalize language
            found_lang = None
            for key, config in LANGUAGE_CONFIG.items():
                if lang in config["aliases"]:
                    found_lang = key
                    break
            
            if found_lang:
                return code, found_lang
        
        # If no known language found, return the last block with its language tag
        last_match = code_matches[-1]
        return last_match.group(2).strip(), last_match.group(1).lower()

    # Look for generic code blocks
    code_match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip(), 'python' # Default to python for generic blocks
    
    # Heuristics for Python
    cleaned_response = response.strip()
    if "def " in cleaned_response or "import " in cleaned_response or "print(" in cleaned_response:
        return cleaned_response, 'python'
        
    # Heuristics for C++
    if "#include" in cleaned_response and ("<iostream>" in cleaned_response or "using namespace std" in cleaned_response):
        return cleaned_response, 'cpp'

    return "", ""

def extract_code(response: str) -> str:
    """
    Extract Python code from the model's response.
    Handles <think> tags and markdown code blocks.
    Wrapper for backward compatibility.
    """
    code, _ = extract_code_with_lang(response)
    return code

def run_command(command: List[str], input_str: str = None, timeout: int = 2) -> Tuple[str, str, int]:
    """
    Run a command with optional input and timeout.
    Returns (stdout, stderr, return_code).
    """
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        try:
            stdout, stderr = process.communicate(input=input_str, timeout=timeout)
            return stdout.strip(), stderr.strip(), process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            return "", "Timeout", -1
            
    except Exception as e:
        return "", str(e), -1

def run_test_case(code_path: str, input_str: str, timeout: int = 2) -> Tuple[str, str, int]:
    """
    Run a single test case against the python script.
    Wrapper for backward compatibility, assumes Python.
    """
    return run_command([sys.executable, code_path], input_str, timeout)

def normalize_output(output: str) -> str:
    """Normalize output for comparison (trim whitespace, handle newlines)."""
    return "\n".join([line.rstrip() for line in output.strip().splitlines()]).strip()

def judge_code_response(response: str, example: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Judge if the code response is correct based on test cases in the example.
    
    Args:
        response: Model's response text
        example: Dataset example containing 'examples' (input/output pairs)
        
    Returns:
        Tuple of (is_correct, extracted_code, execution_info)
    """
    code, lang = extract_code_with_lang(response)
    
    execution_info = {
        "executed": False,
        "code_extracted": bool(code),
        "language": lang,
        "compile_success": None,
        "compile_error": "",
        "test_cases_results": []
    }
    
    if not code:
        return False, "", execution_info
    
    # Default to python if language not supported or unknown
    config = None
    if lang in LANGUAGE_CONFIG:
        config = LANGUAGE_CONFIG[lang]
    else:
        # Try to detect if it's C++ by content if lang is empty or unknown
        if "#include" in code:
            lang = 'cpp'
            config = LANGUAGE_CONFIG['cpp']
        else:
            lang = 'python'
            config = LANGUAGE_CONFIG['python']
            
    # Extract test cases
    test_cases = []
    if 'examples' in example:
        examples_raw = example['examples']
        if isinstance(examples_raw, str):
            try:
                # Try to parse python literal or json
                import ast
                test_cases = ast.literal_eval(examples_raw)
            except:
                pass
        elif isinstance(examples_raw, list):
            test_cases = examples_raw
            
    if not test_cases:
        return False, code, execution_info

    # Create temp directory for compilation artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        # Determine file name
        filename = "solution" + config["extensions"][0]
        if lang == 'java':
            # For Java, we need to match class name. 
            # Simple heuristic: find "public class X"
            match = re.search(r'public\s+class\s+(\w+)', code)
            if match:
                filename = match.group(1) + ".java"
                main_class = match.group(1)
            else:
                filename = "Main.java"
                main_class = "Main"
                # If code doesn't have public class, we might need to wrap it or rename it?
                # Assuming standard competitive programming format where class is Main or Solution
        
        code_path = os.path.join(temp_dir, filename)
        
        with open(code_path, 'w') as f:
            f.write(code)
            
        # Compile if needed
        exe_path = os.path.join(temp_dir, "solution_exe")
        
        if config["compile_cmd"]:
            compile_cmd = [
                arg.format(src=code_path, exe=exe_path, dir=temp_dir) 
                for arg in config["compile_cmd"]
            ]
            stdout, stderr, return_code = run_command(compile_cmd, timeout=30) # Increased timeout for compilation
            
            if return_code != 0:
                execution_info["compile_success"] = False
                execution_info["compile_error"] = stderr + "\n" + stdout
                return False, code, execution_info
            
            execution_info["compile_success"] = True
        
        # Prepare run command
        if config["compile_cmd"]:
            # Compiled
            if lang == 'java':
                run_cmd_template = config["run_cmd"]
                run_cmd = [
                    arg.format(src=code_path, exe=exe_path, dir=temp_dir, main_class=main_class)
                    for arg in run_cmd_template
                ]
            else:
                run_cmd = [exe_path]
        else:
            # Interpreted
            if lang == 'python':
                run_cmd = [sys.executable, code_path]
            else:
                # Fallback for other interpreted languages if added
                run_cmd = config["run_cmd"] + [code_path]

        all_passed = True
        execution_info["executed"] = True
        
        for test_case in test_cases:
            input_val = ""
            output_val = ""
            
            if isinstance(test_case, dict):
                input_val = test_case.get('input', '')
                output_val = test_case.get('output', '')
            elif isinstance(test_case, (list, tuple)) and len(test_case) >= 2:
                input_val = test_case[0]
                output_val = test_case[1]
            else:
                continue
                
            stdout, stderr, return_code = run_command(run_cmd, input_str=input_val)
            
            passed = True
            if return_code != 0:
                passed = False
                all_passed = False
            elif normalize_output(stdout) != normalize_output(output_val):
                passed = False
                all_passed = False
            
            execution_info["test_cases_results"].append({
                "input": input_val,
                "expected_output": output_val,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "passed": passed
            })
            
            if not passed:
                break
                
        return all_passed, code, execution_info
