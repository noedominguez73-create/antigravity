"""
Run pytest and save results to JSON format
"""
import subprocess
import json
import sys

def run_tests():
    """Execute pytest and capture results"""
    print("🧪 Running pytest test suite...")
    print("="*70)
    
    # Run pytest with JSON report
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_api.py", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("="*70)
    print(f"Exit code: {result.returncode}")
    
    # Save to file
    with open("test_output.txt", "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
    
    print("\n✅ Results saved to test_output.txt")
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
