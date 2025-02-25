from get_initial_buildings import main as get_buildings
from filterColumns import main as filter_columns
from encoding import main as encode
from handle_missing_values import main as handle_missing
from keynumbers import main as summary_statistics

def main():
    get_buildings()
    filter_columns()
    encode()
    handle_missing()
    summary_statistics()
    

if __name__=="__main__":
    main()
