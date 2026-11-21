from pathlib import Path

# e.g., if the script is /project/src/scripts/train.py, .parent is /project/src/scripts

names_file = Path(__file__).resolve().parent.parent / "names.txt"
with open(names_file, "r") as f:
    words = f.read().splitlines()
# A str method that splits the big string into a list of substrings on line boundaries (\n, \r, \r\n, etc.)
# and discards the line-break characters themselves

print(words[:10])
