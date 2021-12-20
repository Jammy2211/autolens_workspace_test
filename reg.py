import re

string = "hi_lens_0_hi"

temp = re.compile('lens..')
res = temp.search(string)

print(res.group(0).title())
stop

print(re.match(regex, string))

# initializing string
test_str = 'geeksforgeeks is best for geeks'

# printing original string
print("The original string is : " + str(test_str))

# initializing Substring
sub_str = '..st'

# Wildcard Substring search
# Using re.search()
temp = re.compile(sub_str)
res = temp.search(test_str)

# printing result
print("The substring match is : " + str(res.group(0)))