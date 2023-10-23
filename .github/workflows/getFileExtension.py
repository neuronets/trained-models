import requests
import sys

# Get URL from standard input when the python file is run
url = sys.argv[1]

response = requests.get(url)

# Get the filename from the response headers
content_type = response.headers["Content-Disposition"]
# Get the file extension
extension = content_type.split("=")[1].split(";")[0]

# Return everything after the first "." and without the last quote
extension = extension[1:].strip('"')
extension = extension[extension.find(".") + 1:]

print(extension)