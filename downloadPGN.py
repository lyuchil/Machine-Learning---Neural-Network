
import requests
import sys

if len(sys.argv) != 3 or not sys.argv[1].isnumeric() or not sys.argv[2].isnumeric():
    print("Incorrect format! 2 inputs expected in form 'year' 'month' like:\n"
            "python3 downloadPGN.py 2013 01")
    exit()

name = "lichess_db_standard_rated_{}-{}".format(sys.argv[1], sys.argv[2])

url = "https://database.lichess.org/standard/{}.pgn.zst".format(name)
response = requests.get(url, stream=True)

with open("{}.pgn.zst".format(name), mode="wb") as file:
    for chunk in response.iter_content(chunk_size=10 * 1024):
        file.write(chunk)

print("Success!")