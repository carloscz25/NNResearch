import pickle
import codecs
f = open("netinfo.nfo", "r")
for l in f:
    line = l
    obj = pickle.loads(codecs.decode(line.encode(), "base64"))
    y = 2

y=2