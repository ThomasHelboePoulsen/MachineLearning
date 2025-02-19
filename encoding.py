import pandas as pd
from enum import Enum
import datetime
from dateutil import parser

class FieldType(Enum):
    CODELIST = 1
    AREA = 2
    POINT = 3
    TIMESTAMP = 4
    NUMBER = 5
    UNKNOWN = 6

COLUMNS = {
 'forretningsproces': FieldType.CODELIST,
 'kommunekode': FieldType.CODELIST,
 'registreringFra': FieldType.TIMESTAMP,
 'virkningFra': FieldType.TIMESTAMP,
 'status': FieldType.CODELIST,
 'byg007Bygningsnummer': FieldType.NUMBER,
 'byg021BygningensAnvendelse': FieldType.CODELIST,
 'byg026Opførelsesår': FieldType.NUMBER,
 'byg027OmTilbygningsår': FieldType.NUMBER,
 'byg032YdervæggensMateriale': FieldType.CODELIST,
 'byg033Tagdækningsmateriale': FieldType.CODELIST,
 'byg037KildeTilBygningensMaterialer': FieldType.CODELIST,
 'byg038SamletBygningsareal': FieldType.AREA,
 'byg039BygningensSamledeBoligAreal': FieldType.AREA,
 'byg040BygningensSamledeErhvervsAreal': FieldType.AREA,
 'byg041BebyggetAreal': FieldType.AREA,
 'byg042ArealIndbyggetGarage': FieldType.AREA,
 'byg043ArealIndbyggetCarport': FieldType.AREA,
 'byg044ArealIndbyggetUdhus': FieldType.AREA,
 'byg045ArealIndbyggetUdestueEllerLign': FieldType.AREA,
 'byg046SamletArealAfLukkedeOverdækningerPåBygningen': FieldType.AREA,
 'byg047ArealAfAffaldsrumITerrænniveau': FieldType.AREA,
 'byg048AndetAreal': FieldType.AREA,
 'byg049ArealAfOverdækketAreal': FieldType.AREA,
 'byg050ArealÅbneOverdækningerPåBygningenSamlet': FieldType.AREA,
 'byg051Adgangsareal': FieldType.AREA,
 'byg053BygningsarealerKilde': FieldType.CODELIST,
 'byg054AntalEtager': FieldType.NUMBER,
 'byg055AfvigendeEtager': FieldType.UNKNOWN,
 'byg056Varmeinstallation': FieldType.CODELIST,
 'byg057Opvarmningsmiddel': FieldType.CODELIST,
 'byg058SupplerendeVarme': FieldType.CODELIST,
 'byg069Sikringsrumpladser': FieldType.NUMBER,
 'byg133KildeTilKoordinatsæt': FieldType.CODELIST,
 'byg134KvalitetAfKoordinatsæt': FieldType.CODELIST,
 'byg135SupplerendeOplysningOmKoordinatsæt': FieldType.CODELIST,
 'byg136PlaceringPåSøterritorie': FieldType.CODELIST,
 'byg404Koordinat': FieldType.POINT,
 'byg406Koordinatsystem': FieldType.CODELIST
}

def main():
    df = pd.read_excel("Data2_ColumunFiltered.xlsx")
    df = one_hot_encode(df)
    df = encode_points(df)
    df = encode_timestamps(df)
    #df = set_null_to_0_when appropriate(df)
    df.to_excel("Data3_encoded.xlsx")

def encode_timestamps(df):
    columns = columns_of_type(COLUMNS,FieldType.TIMESTAMP)
    print("encode timestamps: ",columns)
    for col_name in columns:
        df[f"{col_name}_unix"] = encode_timestamp(df[col_name])
    df.drop(columns=columns,inplace=True)
    return df

def encode_timestamp(column:pd.Series):
    """Return values converted to unix (seconds since 1970 basically)"""
    converted_column = []
    for idx,timestamp_str in column.items():
        timestamp = parser.isoparse(timestamp_str).timestamp()
        converted_column.append(timestamp)
    return converted_column


def encode_points(df):
    """change point columns to 2 columns of easting and northing floats"""
    point_colums = columns_of_type(COLUMNS,FieldType.POINT)
    for col_name in point_colums:
        easting,northing = encode_point(df[col_name])
        df[f"{col_name}_easting"] = easting
        df[f"{col_name}_northing"] = northing
    df.drop(columns=point_colums,inplace=True)
    return df

def encode_point(point_column):
    easting, northing = [],[]
    """return the point encoded as 2 numbers (easting, northing) UTM 32 Euref89
     and discarding the height as it is always 0"""
    for idx,point in point_column.items():
        point = point.split("(")[1]
        point = point.split(")")[0]
        point = point.split(" ")
        easting.append(float(point[0]))
        northing.append(float(point[1]))
    return easting,northing





def one_hot_encode(df):
    one_hot_columns = columns_of_type(COLUMNS,FieldType.CODELIST)
    return pd.get_dummies(df,columns=one_hot_columns, dtype=int)

def columns_of_type(columns,type:FieldType):
    return [col for col,col_type in columns.items() if col_type == type]

if __name__ == "__main__":
    main()
