import re
import numpy as np

filename = "/home/aglover-iit.local/data/move-iit-hpe-subset1/P1/drum_f1/atis-s/data.log"
#filename = "/home/cappe/hpe/move-iit-hpe-subset1/P1/drum_f1/atis-s/data.log"


pattern = re.compile('(\d+) (\d+\.\d+) ([A-Z]+) (\d+) (.+)')
#b'\x80abc'.decode("utf-8", "replace")

def unquoting(match):
    matchedString = (match.string[match.span()[0]:match.span()[1]])
    even = len(matchedString) % 2 == 0
    if matchedString[-2:] == b'\\0':
        return b'\\' * ((len(matchedString) - 1) // 2) + b'\0' if even\
            else b'\\' * ((len(matchedString) - 1) // 2) + b'0'
    if matchedString[-2:] == b'\\n':
        return b'\\' * ((len(matchedString) - 1) // 2) + b'\n' if even\
            else b'\\' * ((len(matchedString) - 1) // 2) + b'n'
    if matchedString[-2:] == b'\\r':
        return b'\\' * ((len(matchedString) - 1) // 2) + b'\r' if even\
            else b'\\' * ((len(matchedString) - 1) // 2) + b'r'
    if matchedString[-2:] == b'\\"':
        return b'\"'
    if matchedString[-2:] == b'\\\\':
        if match.string[match.span()[1]:match.span()[1]+1] in [b'0', b'n', b'r', b'\0', b'\n', b'\r']:
            return matchedString
        return b'\\' * ((len(matchedString)) // 2)

with open(filename, mode='rb') as f: #open the file
    binarydata = f.readline()
    print(binarydata)
    
    print(' ')
    print("Massi's Method")
    #print(binarydata)
    firstQuoteIdx = binarydata.find(b'\"')
    lastQuoteIdx = binarydata[::-1].find(b'\"')
    binaryevents = binarydata[firstQuoteIdx + 1:-(lastQuoteIdx + 1)]
    #print(len(binaryevents))
    binaryevents2 = re.sub(b'\\\{2,}|\\\\\"', unquoting, re.sub(b'\\\\+[nr0]', unquoting, binaryevents))
    print(binaryevents2)
    print(len(binaryevents2))
    #bitStrings = np.frombuffer(binaryevents2, np.uint32)
    #print(bitStrings)

    print(' ')
    print("Arren's Method")
    stringdata = binarydata.decode("utf-8", "ignore")
    #print(stringdata)
    found = pattern.match(stringdata)

    #print(found[1])
    #print(found[2])
    #print(found[3])
    #print(found[4])
    #print(found[5])
    data = found[5][1:-1]
    print(data.encode('ascii', 'ignore'))
    print(len(data))

    #print(np.frombuffer(data, np.uint32))

    #print(ord(data[0]))
    #data = np.uint32(data.encode('ascii'))

exit()

with open(filename, mode='rb') as f: #open the file
    
    d = ""
    c = f.read(1).decode("utf-8")
    while( c != ' '):
        d = d + c
        c = f.read(1).decode("utf-8")
    print(int(d))

    d = ""
    c = f.read(1).decode("utf-8")
    while( c != ' '):
        d = d + c
        c = f.read(1).decode("utf-8")
    print(float(d))

    d = ""
    c = f.read(1).decode("utf-8")
    while( c != ' '):
        d = d + c
        c = f.read(1).decode("utf-8")
    print(d)



    d = ""
    c = f.read(1).decode("utf-8")
    while( c != ' '):
        d = d + c
        c = f.read(1).decode("utf-8")
    print(int(d))

    

    print(f.readline())


    #f.read(
    #found = pattern.match(f.readline())
    #print(found)

    #print(int(f.read(8)))
    #.read(1)
    #print(float(f.read(8)))
    #contents = f.readline() #put the lines to a variable (list).
    #print(contents)
