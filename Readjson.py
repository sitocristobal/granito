import json

with open ('labels_json\json-min.json') as json_file:
    data = json.load(json_file)

    # Cambiar las comillas por comillas dobles
    formatearjson = json.dumps(data)

#print(formatearjson)
print(type(formatearjson))