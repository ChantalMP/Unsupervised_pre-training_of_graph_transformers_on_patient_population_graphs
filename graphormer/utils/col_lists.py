'''
mimic_dems_type = [('age', 'sta_cont'), ('gender', 'sta_disc'),('admission_type', 'sta_disc'),('first_careunit', 'sta_disc')]
mimic_dems = ['age', 'gender','admission_type','first_careunit']
mimic_vals = ["('mean corpuscular hemoglobin', 'mean')", "('anion gap', 'mean')", "('partial pressure of oxygen', 'mean')", "('phosphorous', 'mean')",
              "('positive end-expiratory pressure set', 'mean')", "('sodium', 'mean')", "('calcium', 'mean')", "('glascow coma scale total', 'mean')",
              "('glucose', 'mean')", "('lactic acid', 'mean')", "('cardiac output thermodilution', 'mean')",
              "('partial pressure of carbon dioxide', 'mean')", "('ph', 'mean')", "('tidal volume set', 'mean')",
              "('co2 (etco2, pco2, etc.)', 'mean')", "('respiratory rate', 'mean')", "('temperature', 'mean')",
              "('peak inspiratory pressure', 'mean')", "('pulmonary artery pressure systolic', 'mean')", "('partial thromboplastin time', 'mean')",
              "('mean corpuscular volume', 'mean')", "('potassium serum', 'mean')", "('fraction inspired oxygen', 'mean')", "('magnesium', 'mean')",
              "('plateau pressure', 'mean')", "('prothrombin time pt', 'mean')", "('red blood cell count', 'mean')", "('chloride', 'mean')",
              "('respiratory rate set', 'mean')", "('prothrombin time inr', 'mean')", "('mean blood pressure', 'mean')",
              "('tidal volume observed', 'mean')", "('weight', 'mean')", "('diastolic blood pressure', 'mean')",
              "('tidal volume spontaneous', 'mean')", "('heart rate', 'mean')", "('mean corpuscular hemoglobin concentration', 'mean')",
              "('platelets', 'mean')", "('pulmonary artery pressure mean', 'mean')", "('blood urea nitrogen', 'mean')", "('phosphate', 'mean')",
              "('hematocrit', 'mean')", "('potassium', 'mean')", "('calcium ionized', 'mean')", "('central venous pressure', 'mean')",
              "('white blood cell count', 'mean')", "('creatinine', 'mean')", "('co2', 'mean')", "('systemic vascular resistance', 'mean')",
              "('cardiac index', 'mean')", "('fraction inspired oxygen set', 'mean')", "('oxygen saturation', 'mean')", "('bicarbonate', 'mean')",
              "('systolic blood pressure', 'mean')", "('lactate', 'mean')", "('hemoglobin', 'mean')"]

sepsis_dems_type = [('age', 'sta_cont'), ('gender', 'sta_disc'), ('Unit1', 'sta_disc'), ('Unit2', 'sta_disc'), ('HospAdmTime', 'sta_cont')]
sepsis_dems = ['age', 'gender', 'Unit1', 'Unit2', 'HospAdmTime']
sepsis_vals = ["('heart rate', 'mean')", 'O2Sat', "('temperature', 'mean')", "('systolic blood pressure', 'mean')", 'MAP',
               "('diastolic blood pressure', 'mean')", "('respiratory rate', 'mean')", "('co2 (etco2, pco2, etc.)', 'mean')", 'BaseExcess',
               "('bicarbonate', 'mean')", "('fraction inspired oxygen', 'mean')", "('ph', 'mean')", "('partial pressure of carbon dioxide', 'mean')",
               "('oxygen saturation', 'mean')", 'AST', "('blood urea nitrogen', 'mean')", 'Alkalinephos', "('calcium', 'mean')",
               "('chloride', 'mean')", "('creatinine', 'mean')", 'Bilirubin_direct', "('glucose', 'mean')", "('lactate', 'mean')",
               "('magnesium', 'mean')", "('phosphate', 'mean')", "('potassium', 'mean')", 'Bilirubin_total', 'TroponinI', "('hematocrit', 'mean')",
               "('hemoglobin', 'mean')", "('partial thromboplastin time', 'mean')", 'WBC', 'Fibrinogen', "('platelets', 'mean')"]

all_possible_feats =list(set(mimic_dems_type + [(f, 'ts_cont') for f in mimic_vals] + sepsis_dems_type + [(f, 'ts_cont') for f in sepsis_vals]))
'''

mimic_dems_type = [("age", "sta_cont"), ("gender", "sta_disc"), ("admission_type", "sta_disc"), ("first_careunit", "sta_disc")]
mimic_dems = ["age", "gender", "admission_type", "first_careunit"]
mimic_vals = ["(mean corpuscular hemoglobin, mean)", "(anion gap, mean)", "(partial pressure of oxygen, mean)", "(phosphorous, mean)",
              "(positive end-expiratory pressure set, mean)", "(sodium, mean)", "(calcium, mean)", "(glascow coma scale total, mean)",
              "(glucose, mean)", "(lactic acid, mean)", "(cardiac output thermodilution, mean)",
              "(partial pressure of carbon dioxide, mean)", "(ph, mean)", "(tidal volume set, mean)",
              "(co2 (etco2, pco2, etc), mean)", "(respiratory rate, mean)", "(temperature, mean)",
              "(peak inspiratory pressure, mean)", "(pulmonary artery pressure systolic, mean)", "(partial thromboplastin time, mean)",
              "(mean corpuscular volume, mean)", "(potassium serum, mean)", "(fraction inspired oxygen, mean)", "(magnesium, mean)",
              "(plateau pressure, mean)", "(prothrombin time pt, mean)", "(red blood cell count, mean)", "(chloride, mean)",
              "(respiratory rate set, mean)", "(prothrombin time inr, mean)", "(mean blood pressure, mean)",
              "(tidal volume observed, mean)", "(weight, mean)", "(diastolic blood pressure, mean)",
              "(tidal volume spontaneous, mean)", "(heart rate, mean)", "(mean corpuscular hemoglobin concentration, mean)",
              "(platelets, mean)", "(pulmonary artery pressure mean, mean)", "(blood urea nitrogen, mean)", "(phosphate, mean)",
              "(hematocrit, mean)", "(potassium, mean)", "(calcium ionized, mean)", "(central venous pressure, mean)",
              "(white blood cell count, mean)", "(creatinine, mean)", "(co2, mean)", "(systemic vascular resistance, mean)",
              "(cardiac index, mean)", "(fraction inspired oxygen set, mean)", "(oxygen saturation, mean)", "(bicarbonate, mean)",
              "(systolic blood pressure, mean)", "(lactate, mean)", "(hemoglobin, mean)"]

sepsis_dems_type = [("age", "sta_cont"), ("gender", "sta_disc"), ("Unit1", "sta_disc"), ("Unit2", "sta_disc"), ("HospAdmTime", "sta_cont")]
sepsis_dems = ["age", "gender", "Unit1", "Unit2", "HospAdmTime"]
sepsis_vals = ["(heart rate, mean)", "O2Sat", "(temperature, mean)", "(systolic blood pressure, mean)", "MAP",
               "(diastolic blood pressure, mean)", "(respiratory rate, mean)", "(co2 (etco2, pco2, etc), mean)", "BaseExcess",
               "(bicarbonate, mean)", "(fraction inspired oxygen, mean)", "(ph, mean)", "(partial pressure of carbon dioxide, mean)",
               "(oxygen saturation, mean)", "AST", "(blood urea nitrogen, mean)", "Alkalinephos", "(calcium, mean)",
               "(chloride, mean)", "(creatinine, mean)", "Bilirubin_direct", "(glucose, mean)", "(lactate, mean)",
               "(magnesium, mean)", "(phosphate, mean)", "(potassium, mean)", "Bilirubin_total", "TroponinI", "(hematocrit, mean)",
               "(hemoglobin, mean)", "(partial thromboplastin time, mean)", "WBC", "Fibrinogen", "(platelets, mean)"]

all_possible_feats = list(set(mimic_dems_type + [(f, "ts_cont") for f in mimic_vals] + sepsis_dems_type + [(f, "ts_cont") for f in sepsis_vals]))
