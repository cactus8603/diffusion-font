from pathlib import Path
s = f"123.pth"
path = Path('./123')
string = str(path / s)

print(string)