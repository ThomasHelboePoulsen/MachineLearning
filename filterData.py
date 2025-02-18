import ijson
import pandas as pd

def try_convert(val,type):
   try:
      return type(val)
   except:
      None

kommuner = ["0101","0173"]
count = 0
buildings = []
with open('entire current bbr\BBR_V1_Bygning_TotalDownload_json_Current_176.json', "rb") as f:
    for record in ijson.items(f, "item"):
        status = try_convert(record["status"],int)
        anv = try_convert(record["byg021BygningensAnvendelse"],int)
        kommunekode = try_convert(record["kommunekode"],str)
        if kommunekode is not None:
          kommunekode = kommunekode.rstrip().lstrip()
        if status == 6 and anv == 120 and kommunekode in kommuner:
          buildings.append(record)
          count += 1
          print(count)

data = pd.DataFrame(buildings)
print(data)

data.to_excel("Data1_SelectedRows.xlsx")
