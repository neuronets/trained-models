import os
import re

# Get the python_scripts environment variable
python_scripts = os.environ.get("pythons")

# Split the URLs into a list and strip whitespace
urls = [url.strip() for url in python_scripts.strip().split('\n')]

svn_urls = []
# Loop through each URL
for url in urls:
    # Replace /tree/branchName or /blob/branchName with /trunk
    svn_url = re.sub(r'/tree/[^/]+|/blob/[^/]+', '/trunk', url)
    
    svn_urls.append(svn_url)

# Print the generated SVN URLs separated by newline
print('\n'.join(svn_urls))