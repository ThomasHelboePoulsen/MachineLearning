import pandas as pd

wanted_columns = [
 'forretningsproces',
 'id_lokalId',
 'kommunekode',
 'registreringFra',
 'virkningFra',
 'status',
 'byg007Bygningsnummer',
 'byg021BygningensAnvendelse',
 'byg026Opførelsesår',
 'byg027OmTilbygningsår',
 'byg032YdervæggensMateriale',
 'byg033Tagdækningsmateriale',
 'byg037KildeTilBygningensMaterialer',
 'byg038SamletBygningsareal',
 'byg039BygningensSamledeBoligAreal',
 'byg040BygningensSamledeErhvervsAreal',
 'byg041BebyggetAreal',
 'byg042ArealIndbyggetGarage',
 'byg043ArealIndbyggetCarport',
 'byg044ArealIndbyggetUdhus',
 'byg045ArealIndbyggetUdestueEllerLign',
 'byg046SamletArealAfLukkedeOverdækningerPåBygningen',
 'byg047ArealAfAffaldsrumITerrænniveau',
 'byg048AndetAreal',
 'byg049ArealAfOverdækketAreal',
 'byg050ArealÅbneOverdækningerPåBygningenSamlet',
 'byg051Adgangsareal',
 'byg053BygningsarealerKilde',
 'byg054AntalEtager',
 'byg055AfvigendeEtager',
 'byg056Varmeinstallation',
 'byg057Opvarmningsmiddel',
 'byg058SupplerendeVarme',
 'byg069Sikringsrumpladser',
 'byg133KildeTilKoordinatsæt',
 'byg134KvalitetAfKoordinatsæt',
 'byg135SupplerendeOplysningOmKoordinatsæt',
 'byg136PlaceringPåSøterritorie',
 'byg404Koordinat',
 'byg406Koordinatsystem'
]


df = pd.read_excel("Data1_SelectedRows.xlsx")[wanted_columns]
df.to_excel("Data2_ColumunFiltered.xlsx",index=False)
