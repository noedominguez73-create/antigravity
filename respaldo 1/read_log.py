def read_log():
    try:
        with open('debug_results.log', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:50]:
                print(line.strip())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_log()
