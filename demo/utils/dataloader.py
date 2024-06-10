from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage
import pandas as pd

def load_sunbelt_data(states: list=['AL', 'AZ', 'FL',
                                    'GA', 'LA', 'MS',
                                   # 'NM', 'SC', 'TX', 
                                    'CA']):
    data_source = ACSDataSource(survey_year='2018',
                            horizon='1-Year',
                            survey='person')
                            
    data_sunbelt = data_source.get_data(states=states,
                                        download=True)

    features_income, _, _ = ACSIncome.df_to_pandas(data_sunbelt)
    all_variables = set(list(features_income.columns) + ['PINCP'])
    want_data = data_sunbelt.loc[:, list(all_variables)].dropna()

    repl_dict = { 1: 'white',
                  2: 'black',
                  3: 'native_american',
                  4: 'native_alaskan',
                  5: 'native_both',
                  6: 'asian',
                  7: 'pacific_islander',
                  8: 'other',
                  9: 'mixed'}

    want_data.RAC1P = want_data.RAC1P.map(repl_dict)

    # Filter according to doc
    want_data = want_data.loc[want_data.AGEP >= 18,: ]
    want_data = want_data.loc[want_data.WKHP >= 1,: ]
    want_data = want_data.loc[want_data.PINCP >= 100,: ]

    return want_data

def load_sunbelt_coverage(states: list=['AL', 'AZ', 'FL',
                                    'GA', 'LA', 'MS',
                                    'NM', 'SC', 'TX', 
                                    'CA']):
    data_source = ACSDataSource(survey_year='2018',
                            horizon='1-Year',
                            survey='person')
                            
    data_sunbelt = data_source.get_data(states=states,
                                        download=True)

    features_income, _, _ = ACSPublicCoverage.df_to_pandas(data_sunbelt)
    all_variables = set(list(features_income.columns) + ['PUBCOV'])
    want_data = data_sunbelt.loc[:, list(all_variables)]

    repl_dict = { 1: 'white',
                  2: 'black',
                  3: 'native_american',
                  4: 'native_alaskan',
                  5: 'native_both',
                  6: 'asian',
                  7: 'pacific_islander',
                  8: 'other',
                  9: 'mixed'}

    want_data.RAC1P = want_data.RAC1P.map(repl_dict)

    # Filter according to doc
    want_data = want_data.loc[want_data.PINCP <= 30000,: ]
    want_data = want_data.loc[want_data.AGEP < 65,: ]

    return want_data