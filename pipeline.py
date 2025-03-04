from get_initial_buildings import main as get_buildings
from filterColumns import main as filter_columns
from encoding import main as encode
from handle_missing_values import main as handle_missing
from keynumbers import main as summary_statistics
from remove_constant_columns import main as remove_constant_valued_columns

def main():
    get_buildings()
    filter_columns()
    encode()
    handle_missing()
    summary_statistics()
    remove_constant_valued_columns()


if __name__=="__main__":
    main()
