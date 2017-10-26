# -*- coding: utf-8 -*-
# WITH PERFECT DIAGNOSTIC TEST THERE'S SOME NAUGHTY PATIENTS GOING TO A CSC INCORRECTLY
"""
Created on Tue Oct  3 12:20:06 2017

@author: kp331
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:31:19 2017

@author: kp331
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:57:42 2017

@author: kp331
"""

# To save the time running the code to make the output files for the 2 options (if forced to go to CSC, or if go to their nearest open hospital)
# This code reads in the output files so that can then work on the diagnostic test around it

import numpy as np
import pandas as pd
import os
import function_171023_weighted_percentiles as wp

#THE PERFORMANCE INDICATORS THIS MODEL REPORTS
#TIME TO TREATMENT
# 1. For the LVO patients: Time to thrombolysis (mean, median, quartiles, percentiles)
# 2. For the nonLVO patients: Time to thrombolysis  (mean, median, quartiles, percentiles)
# 3. For the LVO patients: Time to thrombectomy (mean, median, quartiles, percentiles)
#TRANSFERS
# 1. Proportion of LVO patients using onward transfer
# 2. Number of LVO admissions going to right location first
# 3. Number of nonLVO admissions going to right location first
# 4. Number of mimic admissions going to right location first
# 5. Number of haemorrage admissions going to right location first
#ADMISSIONS
# 1. Number going to a CSC - First allocation
# 2. Number going to a HASU - First allocation
# 3. Number going to a CSC - Second allocation
# 4. Number going to a HASU - Second allocation
# 5. Per centre the admissions for first allocation (distribution statistics about centre size to know whether CSCs are too large, or HASUs are too small)
# 6. Per centre the admissions for second allocation (distribution statistics about centre size to know whether CSCs are too large, or HASUs are too small)

#THE MODEL
# STEP 1: User defines the system
# 1. Admissions condition division
LVO_as_proportion_of_ischaemic_stroke = 0.4
mimics_as_rate_of_stroke = 1  # rate of mimics as a % of strokes
haemorragics_as_rate_of_ischaemic_strokes=0.5 #Rate of haemorragics as % of ischaemic strokes

# 2. Patient characteristics
onset_time_known = 0.5  # Stroke symptom onset time known (used to select the patients to do the LVO diagnostic test on, else to HASU as will not be eligible for any treatment)

# 3. Hospital allocation
Decision_bias = 15  # Any patient with a CSC less than this many minutes more (over a nearer HASU) will have their location determined by the diagnostic test.

# 4. Diagnostic test performance
Specificity_stroke = 0.9  # Specificity for stroke patients (used on the nonLVO stroke patients)
Sensitivity_stroke = 0.66  # Sensitivity for stroke patients (used on the LVO stroke patients)
Specificity_mimics = 0.98  # Specificity for mimics (used on the mimics, assume less are likely to be misclassified as a LVO)

# 5. Treatment eligibility
nonLVO_thrombolysis_eligible = 0.1  # nonLVO patients that are suitable for thrombolysis
nonLVO_known_onsettime_thrombolysis_eligible = 0.2  # LVO patients that are suitable for thrombolysis
LVO_thrombolysis_eligible = 0.35  # LVO patients that are suitable for thrombolysis
LVO_known_onsettime_thrombolysis_eligible = 0.7  # LVO patients that are suitable for thrombolysis
LVO_thrombolysis_thrombectomy_eligible = 0.77  # Of those LVO patients suitable for thrombolysis, proportion suitable for thrombectomy
# REPLACES Thrombectomy_eligible=0.27

# 6. Repatriation(for those patients at a CSC with a more local HASU and not requiring thrombectomy)
repatriation_thrombolysis = 0.8  # Proportion patients had thrombolysis that are repatriated
repatriation_notthrombolysis = 0.9  # Proportion patients not had thrombolysis that are repatriated

#Calculate from these user defined metrics the parameters for the model
mimics = (mimics_as_rate_of_stroke * (1 / (1 + mimics_as_rate_of_stroke)))
strokes=(1-mimics)
ischaemic=1/(mimics_as_rate_of_stroke+1)/(haemorragics_as_rate_of_ischaemic_strokes+1)
haemorragics=(haemorragics_as_rate_of_ischaemic_strokes*(1/(1+haemorragics_as_rate_of_ischaemic_strokes)))*(1-mimics)
LVO = LVO_as_proportion_of_ischaemic_stroke * ischaemic#was: LVO = LVO_as_proportion_of_stroke * (1 / (1 + mimics_as_rate_of_stroke))
nonLVO = (1 - LVO_as_proportion_of_ischaemic_stroke) * ischaemic#was: nonLVO = (1 - LVO_as_proportion_of_stroke) * (1 / (1 + mimics_as_rate_of_stroke))

# Patients will fall into 3 categories 1) CSC is nearest so go there 2) CSC is within extra n mins more than local HASU so use diagnostic test 3) CSC is more than n mins more than local HASU so go to HASU
# set Decision_bias to a high value (eg. 99999) for all patients who don't live nearest to a CSC to get their location determined by the diagnostic test and so no LSOA will go automatically to a HASU

# STEP 2: Calculate the two options available for each LSOA
# Option i. If all patients must all go to their nearest CSC [a CSC is defined by a "2" in the population array]
# Option ii. If patients can go to any open centre [a HASU is defined by a "1" and a CSC is defined by a "2" in the population array, so just not a centre defined as a "0" as is shut]

# Set output location for the files to be written to
OUTPUT_LOCATION = 'output/Version1'
# Create new folder if folder does not already exist
if not os.path.exists(OUTPUT_LOCATION):
    os.makedirs(OUTPUT_LOCATION)

# Score_matrix:
# 0: Number of hospitals
# 1: Average distance
# 2: Maximum distancxcsd se
# 3: Maximum admissions to any one hopsital
# 4: Minimum admissions to any one hopsital
# 5: Max/Min Admissions ratio
# 6: Proportion patients within target distance 1
# 7: Proportion patients within target distance 2
# 8: Proportion patients within target distance 3
# 9: Proportion patients attending unit with target admission numbers
# 10: Proportion of patients meeting distance 1 and admissions target
# 11: Proportion of patients meeting distance 2 and admissions target
# 12: Proportion of patients meeting distance 3 and admissions target
# 13: Clinical benefit if thrombolysis (fixed door to needle = mins + fixed onset to travelling in ambulance time = mins + travel time which is model dependant).  Additional benefit per 100 treatable patients
#output_array = np.empty((19, 1))
nscore_parameters = 13

HOSPITALS = pd.read_csv(
    'data/hospitals.csv')  # Read hospital info into dataframe.  Contains hospital name, easting & northing
HOSPITAL_COUNT = len(HOSPITALS.index)
population = np.zeros((0, HOSPITAL_COUNT))

## Import data
TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS_60 = np.loadtxt('data/inter_hospital_time_plus60_no_index.csv',
                                                 delimiter=',')  # Node to hospital matrix.  PREVIOUSLY using "msoa_truncated_dist_matrix.csv"
#TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS = np.loadtxt('data/inter_hospital_time_no_index.csv', delimiter=',')  # Node to hospital matrix.  PREVIOUSLY using "msoa_truncated_dist_matrix.csv"

# TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS=np.loadtxt('data/inter_hospital_time_plus60_no_index.csv',delimiter=',') # Node to hospital matrix.  PREVIOUSLY using "msoa_truncated_dist_matrix.csv"
# population=np.loadtxt('data/170822_Load_CSC_DripNShip_solutions_3_5_10.csv',delimiter=',')
population = np.loadtxt('data/170822_Load_LVO_solution_5.csv',
                        delimiter=',')  # Contains 1=HASU and 2=CSC and 0 = closed
if len(
        population.shape) == 1:  # return the number of dimensions.  if only 1 population then the shape will be 1 dimension and so need to add a dimension so that it is in a standard format for when there are more than one solution.
    population = np.array([population])

# Using population to set up Option i. If all patients must all go to their nearest CSC [defined by a "2" in the population array]
# Solution for patients to go to just CSC: only CSC open (=1).  Rest shut (=0)
population_MS = np.copy(population)
population_MS[population_MS == 1] = 0
population_MS[population_MS == 2] = 1
mask_csc = np.array(population_MS, dtype=bool)
mask_csc = mask_csc.flatten()

# Using population to set up Option ii. If can go to any open centre [defined by a "1" or a "2" in the population array]
# Solution for patients to go to any open centre: both CSC & HASUs open (=1).  Rest shut (=0)
population_All = np.copy(population)
population_All[population_All == 2] = 1
mask_soln_hosp = np.array(population_All, dtype=bool)
mask_soln_hosp = mask_soln_hosp.flatten()

ADMISSIONS = np.loadtxt('data/LSOA_strokes_no_index.csv',
                        delimiter=',')  # Admissions for each patient node.  PREVIOUSLY using "msoa_truncated_stemi.csv"

LVO_ADMISSIONS=ADMISSIONS*LVO
nonLVO_ADMISSIONS=ADMISSIONS*nonLVO
mimic_ADMISSIONS=ADMISSIONS*mimics
haemorrage_ADMISSIONS=ADMISSIONS*haemorragics

# ADMISSIONS=np.loadtxt('data/LSOA_strokes_to_London_NSCs_no_index.csv',delimiter=',') # Admissions for each patient node.  PREVIOUSLY using "msoa_truncated_stemi.csv"
TOTAL_ADMISSIONS = np.sum(ADMISSIONS)
TRAVEL_MATRIX = np.loadtxt('data/stroke_time_matrix_data_only.csv',
                           delimiter=',')  # Node to hospital matrix.  PREVIOUSLY using "msoa_truncated_dist_matrix.csv"

## Initialise variables
TARGET_ADMISSIONS = 1000

# population_size=len(population[:,0]) # current population size. Number of children generated will match this.
population_size = len(population)  # current population size. Number of children generated will match this.

# WOULD HAVE CODE HERE TO CALL THE FUNCTIONS, INSTEAD READ IT IN AND WORK ON THE DATA BELOW.
#Returns the 2 options for each LSOA: 1) going to their nearest open centre 2) going to their nearest HASU
hospital_admissions_matrix_All = np.loadtxt(OUTPUT_LOCATION + '/admissions_All.csv', delimiter=',')
patient_travel_matrix_All = np.loadtxt(OUTPUT_LOCATION + '/patient_travel_All.csv', delimiter=',')
patient_attend_hospital_All = np.loadtxt(OUTPUT_LOCATION + '/patient_attend_hospital_All.csv', delimiter=',')
hospital_admissions_matrix_MS = np.loadtxt(OUTPUT_LOCATION + '/admissions_MS.csv', delimiter=',')
patient_travel_matrix_MS = np.loadtxt(OUTPUT_LOCATION + '/patient_travel_MS.csv', delimiter=',')
patient_attend_hospital_MS = np.loadtxt(OUTPUT_LOCATION + '/patient_attend_hospital_MS.csv', delimiter=',')

ADMISSIONS = np.loadtxt('data/LSOA_strokes_no_index.csv',
                        delimiter=',')  # Admissions for each patient node.  PREVIOUSLY using "msoa_truncated_stemi.csv"

# STEP 2: Analyse the system given the user defined diagnostic test

# CREATE MASKS TO DIVIDE PATIENT LSOAs INTO THOSE THAT GO TO CSC, GO TO HASU OR NEED DIAGNOSTIC TEST TO DECIDE
# Mask for the patients that live closest to a CSC.  These are going to the right place as their nearest location has every treatment
# mask_nearest_hospital_is_csc=patient_attend_hospital_MS==patient_attend_hospital_All

# Mask for the patients that live closest to a HASU.  These patients may not necessarily go to the right location as depends on the diagnostic test as to whether they get sent to HASU or CSC
# mask_nearest_hospital_is_hasu=np.invert(mask_nearest_hospital_is_csc)

# Mask for patients that the difference between CSC and HASU is less than 15 mins.  Only these patients will have the diagnostic test to determine where they go.  Those that are greater than 15 mins extra to get to the CSC will always go to their nearest place (HASU)
# mask_CSC_extra_travel_less_than_bias=(patient_travel_matrix_MS-patient_travel_matrix_All)<=Decision_bias

# Patients live closest to CSC
mask_go_to_CSC = (patient_travel_matrix_MS - patient_travel_matrix_All) == 0

# Extra distance to CSC is too great to consider, go to HASU
mask_go_to_HASU = (patient_travel_matrix_MS - patient_travel_matrix_All) > Decision_bias

# Extra distance to CSC is worth considering, use diagnostic test
mask_use_diagnostic = ((patient_travel_matrix_MS - patient_travel_matrix_All) <= Decision_bias) & (
(patient_travel_matrix_MS - patient_travel_matrix_All) > 0)

# Divide the LSOA admissions into MANY arrays, where the sum of these arrays = 100% of admissions.
# Admissions are divided into mimics & strokes (LVO, nonLVO, haemorrage) and their destination (HASU stay, CSC stay, HASU onwards to CSC, CSC onwards to HASU).
# From each LSOA there are these options
    # Option i) Live closest to CSC, so all go there (and stay there)
    # Option ii) Extra distance to CSC is too great to consider to be the first location, so all go to HASU (Eligible thrombectomy LVO move to CSC)
    # Option iii) Extra distance to CSC is worth considering, use diagnostic test to decide the first location (those in the wrong location can move [CSC to HASU dependant on repatriation rate, & HASU to CSC for thrombectomy]

# ADMISSIONS LIVE CLOSEST TO A CSC SO GO THERE.  THESE PATIENTS ARE ALL TAKEN TO THE RIGHT PLACE
ALL_ADMISSIONS_nearestCSC = ADMISSIONS * mask_go_to_CSC  # mask_nearest_hospital_is_csc

# Divide these into mimics, LVO & nonLVO admissions for performace calculation.
# All these patients get their necessary treatment the quickest
# No repatriation necessary as in their closest hospital

# Mimics that go to CSC first as nearest.  Non treatment for all
mimic_ADMISSIONS_nearestCSC = ALL_ADMISSIONS_nearestCSC * mimics#Array1

# Haemorragic stroke patients that go to CSC first as nearest.  Non treatment for all
haemorrage_ADMISSIONS_nearestCSC = ALL_ADMISSIONS_nearestCSC * haemorragics#Array1

# nonLVO that go to CSC first as nearest
nonLVO_ADMISSIONS_nearestCSC = ALL_ADMISSIONS_nearestCSC * nonLVO #Array2
# divide the nonLVO admissions into the 2 eligible treatment groups: noTreatment, tlysis
nonLVO_ADMISSIONS_nearestCSC_noTreatment = nonLVO_ADMISSIONS_nearestCSC * (1 - nonLVO_thrombolysis_eligible)
nonLVO_ADMISSIONS_nearestCSC_tlysis = nonLVO_ADMISSIONS_nearestCSC * nonLVO_thrombolysis_eligible

# LVO that go to CSC first as nearest
LVO_ADMISSIONS_nearestCSC = ALL_ADMISSIONS_nearestCSC * LVO  # Array3
# divide the LVO admissions into the 3 eligible treatment groups: noTreatment, thrombolysis only, both treatments (thrombolysis and thrombectomy)
LVO_ADMISSIONS_nearestCSC_noTreatment = LVO_ADMISSIONS_nearestCSC * (1 - LVO_thrombolysis_eligible)
LVO_ADMISSIONS_nearestCSC_tlysis = LVO_ADMISSIONS_nearestCSC * LVO_thrombolysis_eligible
LVO_ADMISSIONS_nearestCSC_bothTreatments = LVO_ADMISSIONS_nearestCSC_tlysis * LVO_thrombolysis_thrombectomy_eligible
LVO_ADMISSIONS_nearestCSC_tlysisOnly = LVO_ADMISSIONS_nearestCSC_tlysis * (1 - LVO_thrombolysis_thrombectomy_eligible)

# ADMISSIONS LIVE TOO MUCH FURTHER FROM A CSC SO GO TO A HASU
ALL_ADMISSIONS_nearestHASU = ADMISSIONS * mask_go_to_HASU  # mask_nearest_hospital_is_hasu

# Divide these into mimics, LVO & nonLVO admissions for performace calculation.
# All these patients get their thrombolysis the quickest
# All thrombectomy patients are delayed as they need an onwards transfer
# No repatriation necessary as only those needing thrombectomy are in a centes that's not their nearest

# Mimics that go to HASU first as CSC too far to consider
mimic_ADMISSIONS_nearestHASU = ALL_ADMISSIONS_nearestHASU * mimics  # Array4

# Haemorragic stroke patients that go to HASU first as CSC too far to consider
haemorrage_ADMISSIONS_nearestHASU = ALL_ADMISSIONS_nearestHASU * haemorragics#Array1

# nonLVOs that go to HASU first as CSC too far to consider
nonLVO_ADMISSIONS_nearestHASU = ALL_ADMISSIONS_nearestHASU * nonLVO  # Array5
# divide the nonLVO admissions into the 3 eligible treatment groups: none, thrombolysis, thrombectomy
nonLVO_ADMISSIONS_nearestHASU_noTreatment = nonLVO_ADMISSIONS_nearestHASU * (
1 - nonLVO_thrombolysis_eligible)
nonLVO_ADMISSIONS_nearestHASU_tlysis = nonLVO_ADMISSIONS_nearestHASU * nonLVO_thrombolysis_eligible

# LVOs that go to HASU first as CSC too far to consider
LVO_ADMISSIONS_nearestHASU = ALL_ADMISSIONS_nearestHASU * LVO  # Array6
# divide the LVO admissions into the 3 eligible treatment groups: none, thrombolysis, thrombectomy
LVO_ADMISSIONS_nearestHASU_noTreatment = LVO_ADMISSIONS_nearestHASU * (
1 - LVO_thrombolysis_eligible)
LVO_ADMISSIONS_nearestHASU_tlysis = LVO_ADMISSIONS_nearestHASU * LVO_thrombolysis_eligible
LVO_ADMISSIONS_nearestHASU_bothTreatments = LVO_ADMISSIONS_nearestHASU_tlysis * LVO_thrombolysis_thrombectomy_eligible  # these admissions need a transfer to the CSC
LVO_ADMISSIONS_nearestHASU_tlysisOnly = LVO_ADMISSIONS_nearestHASU_tlysis * (
1 - LVO_thrombolysis_thrombectomy_eligible)  # these admissions need a transfer to the CSC
# LVO_ADMISSIONS_nearestHASU_onwards_to_CSC=LVO_ADMISSIONS_nearestHASU*Thrombectomy_eligible#Array5 go onto the CSC (those eligible for thrombectomy).  This is a replica of admissions (don't sum with the above else will double count these patients)

# ADMISSIONS LIVE WITHIN DISTANCE TO A CSC, USE DIAGNOSTIC TEST TO DETERMINE LOCATION
# Depending on the test's performance there is a mix aobut whether patients get their thrombolysis the quickest
# Depending on the test's performance there is a mix aobut whether patients get their thrombectomy the quickest
# Depending on the test's performance repatriation may be necessary as patients couyld be taken to a cente that's further than their closest
ALL_ADMISSIONS_useDiagnosticTest = ADMISSIONS * mask_use_diagnostic

# Divide these into stroke onset time known, and not.  Only use the diagnostic test on those with a known stroke onset time.
# Those with unknown stroke onset time go to HASU and have no treatment
ALL_ADMISSIONS_useDiagnosticTest_onsettimeKnown = ALL_ADMISSIONS_useDiagnosticTest * onset_time_known
ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown = ALL_ADMISSIONS_useDiagnosticTest * (1 - onset_time_known)


# Divide the known onset time admissions into mimics, LVO & nonLVO admissions for performace calculation.

# MIMICS WITH TIME KNOWN
mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown = ALL_ADMISSIONS_useDiagnosticTest_onsettimeKnown * mimics
# divide by the test and by repatriation
# these admissions are incorrectly diagnosed and sent to further place
mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC = (mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                    (1 - Specificity_mimics))
# these admissions are incorrectly diagnosed and sent to further place and stay there
mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay = (mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                         (1 - Specificity_mimics) *
                                                                         (1 - repatriation_notthrombolysis))
# these admissions are incorrectly diagnosed and sent to further place and repatriated to HASU
mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat = (mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                          (1 - Specificity_mimics) *
                                                                          repatriation_notthrombolysis)
# these admissions are sent to correct place
mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU = mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown * Specificity_mimics
# Mimics that go to HASU as are correctly diagnosed as nonLVO
# mimic_ADMISSIONS_sent_to_HASU_correct=ADMISSIONS_useDiagnosticTest*mimics#Array4

# HAEMORRAGICS WITH TIME KNOWN
haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown = ALL_ADMISSIONS_useDiagnosticTest_onsettimeKnown * haemorragics
#Incorrectly diagnosed (these admissions are incorrectly diagnosed and sent to further place)
haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC = (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                         (1 - Specificity_stroke))
# stay
haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay = (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                              (1 - Specificity_stroke) *
                                                                              (1 - repatriation_notthrombolysis))
# repatriated to HASU
haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat = (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                               (1 - Specificity_stroke) *
                                                                               repatriation_notthrombolysis)
# Correctly diagnosed, these admissions are sent to correct place
haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU = (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                             Specificity_stroke)
# Haemorrages that go to HASU as are correctly diagnosed as nonLVO

# nonLVOs WITH ONSET TIME KNOWN
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown = ALL_ADMISSIONS_useDiagnosticTest_onsettimeKnown * nonLVO
# divide by the test diagnosis, eligibility for treatments and by repatriation
# NONLVOs INCORRECTLY DIAGNOSED
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                     (1 - Specificity_stroke))
# these admissions are incorrectly diagnosed and sent to further place and stay there
# Elig for thrombolysis (But travelled further for it)
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                            (1 - Specificity_stroke) *
                                                                            (nonLVO_known_onsettime_thrombolysis_eligible))
#Stay
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Stay = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                 (1 - Specificity_stroke) *
                                                                                 (nonLVO_known_onsettime_thrombolysis_eligible) *
                                                                                 (1 - repatriation_thrombolysis))
# Repatriate
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Repat = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                  (1 - Specificity_stroke) *
                                                                                  (nonLVO_known_onsettime_thrombolysis_eligible) *
                                                                                  (repatriation_thrombolysis))
# Not elig for thrombolysis (But travelled further for no treatment)
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                 (1 - Specificity_stroke) *
                                                                                 (1 - nonLVO_known_onsettime_thrombolysis_eligible))
# Stay
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                      (1 - Specificity_stroke) *
                                                                                      (1 - nonLVO_known_onsettime_thrombolysis_eligible) *
                                                                                      (1 - repatriation_notthrombolysis))
# Repatriate
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                       (1 - Specificity_stroke) *
                                                                                       (1 - nonLVO_known_onsettime_thrombolysis_eligible) *
                                                                                       (repatriation_notthrombolysis))

# nonLVOs WITH ONSET TIME KNOWN AND CORRECTLY DIAGNOSED
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU = nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown * Specificity_stroke

# these admissions are correctly diagnosed and sent to HASU, elig for thrombolysis.  Got treatment quickest
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysis = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                Specificity_stroke *
                                                                                nonLVO_known_onsettime_thrombolysis_eligible)
# these admissions are correctly diagnosed and sent to HASU, not elig for thrombolysis so no treatment.
nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                     Specificity_stroke *
                                                                                     (1 - nonLVO_known_onsettime_thrombolysis_eligible))

# LVOs WITH ONSET TIME KNOWN
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown = ALL_ADMISSIONS_useDiagnosticTest_onsettimeKnown * LVO

# divide by the test diagnosis, eligibility for treatments and by repatriation

# LVOs WITH ONSET TIME KNOWN AND INCORRECTLY DIAGNOSED
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                      (1 - Sensitivity_stroke))

# these admissions are incorrectly diagnosed and sent to nearer HASU, elig for thrombolysis, not elig for thrombectomy.  Correct place for necessary treatment
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                 (1 - Sensitivity_stroke) *
                                                                                 (LVO_known_onsettime_thrombolysis_eligible) *
                                                                                 (1 - LVO_thrombolysis_thrombectomy_eligible))

# these admissions are incorrectly diagnosed and sent to nearer HASU, elig for thrombolysis, elig for thrombectomy.  Incorrect place for thrombectomy - need to travel onwards
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                     (1 - Sensitivity_stroke) *
                                                                                     (LVO_known_onsettime_thrombolysis_eligible) *
                                                                                     (LVO_thrombolysis_thrombectomy_eligible))
# these admissions are incorrectly diagnosed and sent to nearer HASU, not elig for thrombolysis.  Travelled correct place as no treatment needed
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                  (1 - Sensitivity_stroke) *
                                                                                  (1 - LVO_known_onsettime_thrombolysis_eligible))

# LVOs WITH ONSET TIME KNOWN AND CORRECTLY DIAGNOSED
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                 Sensitivity_stroke)
# these admissions are correctly diagnosed and sent further to CSC, elig for thrombolysis, not elig for thrombectomy - wrong location.
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                             Sensitivity_stroke *
                                                                             LVO_known_onsettime_thrombolysis_eligible *
                                                                             (1 - LVO_thrombolysis_thrombectomy_eligible))
# Repatriate
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Repat = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                   Sensitivity_stroke *
                                                                                   LVO_known_onsettime_thrombolysis_eligible *
                                                                                   (1 - LVO_thrombolysis_thrombectomy_eligible) *
                                                                                   (repatriation_thrombolysis))
# Stay
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Stay = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                  Sensitivity_stroke *
                                                                                  LVO_known_onsettime_thrombolysis_eligible *
                                                                                  (1 - LVO_thrombolysis_thrombectomy_eligible) *
                                                                                  (1 - repatriation_thrombolysis))
# these admissions are correctly diagnosed and sent further to CSC, elig for thrombolysis, elig for thrombectomy - correct location.
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                 Sensitivity_stroke *
                                                                                 LVO_known_onsettime_thrombolysis_eligible *
                                                                                 LVO_thrombolysis_thrombectomy_eligible)

# these admissions are correctly diagnosed and sent further to CSC, not elig for thrombolysis, not elig for thrombectomy - wrong location.
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                              Sensitivity_stroke *
                                                                              (1 - LVO_known_onsettime_thrombolysis_eligible))
#Repatriate
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                    Sensitivity_stroke *
                                                                                    (1 - LVO_known_onsettime_thrombolysis_eligible) *
                                                                                    (repatriation_notthrombolysis))
#Stay
LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay = (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown *
                                                                                   Sensitivity_stroke *
                                                                                   (1 - LVO_known_onsettime_thrombolysis_eligible) *
                                                                                   (1 - repatriation_notthrombolysis))

# check got all the admissions divvied up
sum_admissions = np.sum(ALL_ADMISSIONS_useDiagnosticTest + ALL_ADMISSIONS_nearestCSC + ALL_ADMISSIONS_nearestHASU)
print("Admissions divvied into locations and resummed: ", sum_admissions, " ", np.sum(ADMISSIONS))

# check got all the CSC admissions
sum_admissions = np.sum(haemorrage_ADMISSIONS_nearestCSC+mimic_ADMISSIONS_nearestCSC +
                            nonLVO_ADMISSIONS_nearestCSC_noTreatment +
                            nonLVO_ADMISSIONS_nearestCSC_tlysis +
                            LVO_ADMISSIONS_nearestCSC_noTreatment +
                            LVO_ADMISSIONS_nearestCSC_tlysisOnly +
                            LVO_ADMISSIONS_nearestCSC_bothTreatments)

print("Check got all CSC admissions:", np.sum(ALL_ADMISSIONS_nearestCSC), " ", sum_admissions)

# check got all the CSC admissions
sum_admissions = np.sum(haemorrage_ADMISSIONS_nearestCSC+mimic_ADMISSIONS_nearestCSC + nonLVO_ADMISSIONS_nearestCSC + LVO_ADMISSIONS_nearestCSC)
print("Check got all CSC admissions, again:", np.sum(ALL_ADMISSIONS_nearestCSC), " ", sum_admissions)

# check got all the diagnostic admissions
sum_admissions = np.sum(ALL_ADMISSIONS_useDiagnosticTest_onsettimeKnown + ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown)
print("Check got all diagnostic admissions:", np.sum(ALL_ADMISSIONS_useDiagnosticTest), " ", sum_admissions)

#CALCULATE THE TRAVEL

# for each patient node want the distance from their nearest HASU to the onwards nearest CSC.
# Use the admissions (and so 0 if none) to only select for those that need this distance
# This is for any patient that is LVO&eligible Thrombectomy patient that went to HASU first due to
# 1) CSC was too far [LVO_ADMISSIONS_nearestHASU_ttmoy]
# 2) Diagnostic test incorrect [LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments]
# mask the columns of the travel matrix so patients only choose their nearest CSC
masked_BETWEEN_TRAVEL_MATRIX_60 = TRAVEL_MATRIX_BETWEEN_ALL_HOSPITALS_60[:, mask_csc]
# masked_BETWEEN_TRAVEL_MATRIX=masked_BETWEEN_TRAVEL_MATRIX[mask_soln_hosp,:]
csc_id_from_solution_hospitals = np.asarray(np.where(mask_csc == True))
# full_hosp_ID=np.arange(1,len(TRAVEL_MATRIX[0,:])+1,1)
# csc_id_from_solution_hospitals =full_hosp_ID[mask_csc]

TIME_csc_nearest_to_HASU = np.empty(0)
ID_csc_nearest_to_HASU = np.empty(0)
for n in range(len(masked_BETWEEN_TRAVEL_MATRIX_60)):  # through each hospital and find nearest CSC
    TIME_csc_nearest_to_HASU = np.append(TIME_csc_nearest_to_HASU, np.amin(masked_BETWEEN_TRAVEL_MATRIX_60[n, :]))
    ID_csc_nearest_to_HASU = np.append(ID_csc_nearest_to_HASU, csc_id_from_solution_hospitals[0, np.argmin(masked_BETWEEN_TRAVEL_MATRIX_60[n,:])])  # store the CSC ID (from the hospital solution list) that has the shortest distance from the first hosptial

# HAVE NOW CALCULATED THE NEAREST CSC FROM EACH HASU (ONLY NEED TO DO ONCE) AND SO NOW CAN LOOK IT UP
# patient_node_HASU_CSC_time=TIME_csc_nearest_to_HASU[np.int_(gone_to_HASU_id)]
# patient_node_HASU_CSC_id=ID_csc_nearest_to_HASU[np.int_(gone_to_HASU_id)]
patient_node_HASU_CSC_time = TIME_csc_nearest_to_HASU[np.int_(patient_attend_hospital_All - 1)]  # need -1 as they are ID=1 to 127... when used to array position need 0 to 126
patient_node_HASU_CSC_id = ID_csc_nearest_to_HASU[np.int_(patient_attend_hospital_All - 1)]  # need -1 as they are ID=1 to 127... when used to array position need 0 to 126

# For the admissions that go to a HASU but needed a CSC
patient_travel_matrix_to_HASU_onwards_CSC = (patient_travel_matrix_All + patient_node_HASU_CSC_time)#includes 60 minute turn around at first centre
patient_travel_matrix_to_CSC_onwards_HASU = (patient_travel_matrix_MS + patient_node_HASU_CSC_time)#includes 60 minute turn around at first centre

# Sum these per node, : LVO_ADMISSIONS_nearestHASU_ttmoy,LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments
# and multiply by "patient_travel_matric_to_HASU_onwards_CSC" to get the time patient spends in transit as need thrombectomy and gone to wrong place


#CALCULATE THE PERFORMANCE INDICATORS
output_array = np.zeros((25,25))

#TIME TO TREATMENT
# 1. For the LVO patients: Time to thrombolysis (mean, median, quartiles, percentiles)
# 2. For the nonLVO patients: Time to thrombolysis  (mean, median, quartiles, percentiles)
# 3. For all thrombolysis patients:  Time to thrombolysis (mean, median, quartiles, percentiles)
# 4. For the LVO patients: Time to thrombectomy (mean, median, quartiles, percentiles)
#TRANSFERS
# 1. Proportion of LVO patients using onward transfer
# 2. Number of LVO admissions going to right location first
# 3. Number of nonLVO admissions going to right location first
# 4. Number of mimic admissions going to right location first
# 5. Number of haemorrage admissions going to right location first
# 6. Admissions going to HASU first that's correct
# 7. Admissions going to HASU first that's incorrect
# 8. Admissions going to CSC first that's correct
# 9. Admissions going to CSC first that's incorrect
#ADMISSIONS
# 1. Number going to a CSC - First allocation [admissions_go_to_CSC_first]
# 2. Number going to a HASU - First allocation [admissions_go_to_HASU_first]
# 3. Number going to a CSC - Second allocation [admissions_go_to_CSC_final]
# 4. Number going to a HASU - Second allocation [admissions_go_to_HASU_final]
# 5. Per centre the admissions for first allocation (distribution statistics about centre size to know whether CSCs are too large, or HASUs are too small)
# 6. Per centre the admissions for second allocation (distribution statistics about centre size to know whether CSCs are too large, or HASUs are too small)

#Total minutes travelling if all patients went to correct location first (first array section are those to go to CSC [= those that live nearest plus those LVOs needing thrombectomy], the second array are those to go to HASU [=those that nearest to HASU and not need a thrombectomy)
admissions_array=np.append((ALL_ADMISSIONS_nearestCSC+LVO_ADMISSIONS_nearestHASU_bothTreatments+
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments+
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments),
                           (ALL_ADMISSIONS_nearestHASU-LVO_ADMISSIONS_nearestHASU_bothTreatments+
                            ALL_ADMISSIONS_useDiagnosticTest-
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments-
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments))
time_array=np.append(patient_travel_matrix_MS,patient_travel_matrix_All)
#Time travelled with patients going where they need to first (effectively a 99999 decision bias and perfect diagnostic test for thrombectomy) --> total_time_travel_perfect
total_time_travel_perfectAllocation=np.sum(admissions_array*time_array)
print("Check have all the admissions for the [total_time_travel_perfectAllocation] array: ", np.sum(admissions_array))

#Total minutes travelling with a drip 'n ship model using no diagnostic test --> total_time_travel_noDiag using the same definition of the system as set up for diag test (decicion bias)
#Effectively the difference is that all the patietns getting diagnostic test to decide the location, they all go to CSC and not divied up.

# first array section are those to go to CSC and stay [= those that live nearest plus any diagnostic test sends there and not repatriated],
# the second array are those to go to CSC and repatriated to HASU
# third array are those that go to HASU and stay
# fourth array are those that go to HASU and onward to CSC [=LVO eligible for thrombectomy but went to HASU as either live too far from CSC or diagnostic test incorrectly sent them to HASU)

#np.append() can only take 2 arrays, but want to append 4 so do in 3 steps:

# first array section are those to go to CSC and stay [= those that live nearest plus any patient in the decision bias zone, and not repatriated (none are repatriated who live nearest to CSC, those that are in decision bias zone use the repatriation rate to decide],
# the second array are those to go to CSC and repatriated to HASU  [= those that live nearest plus any patient in the decision bias zone, and repatriated (those that live nearer to HASU use the repatriation rate to decide],
admissions_array=np.append((ALL_ADMISSIONS_nearestCSC +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Stay +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +
                            nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +
                            nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Stay +
                            mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay +
                            haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay +
                            (ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown * (1-repatriation_notthrombolysis)) +
                            (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly  * (1-repatriation_thrombolysis)) +
                            (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment  * (1-repatriation_notthrombolysis)) +
                            (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU * (nonLVO_known_onsettime_thrombolysis_eligible) * (1-repatriation_thrombolysis)) +
                            (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU * (1-nonLVO_known_onsettime_thrombolysis_eligible) * (1-repatriation_notthrombolysis)) +
                            (mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU  * (1-repatriation_notthrombolysis)) +
                            (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU * (1-repatriation_notthrombolysis)) +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments),
                           (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Repat +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat +
                            nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat +
                            nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Repat +
                            mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat +
                            haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat) +
                            (ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown * (repatriation_notthrombolysis)) +
                            (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly * (repatriation_thrombolysis)) +
                            (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment * (repatriation_notthrombolysis)) +
                            (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU * (nonLVO_known_onsettime_thrombolysis_eligible) * (repatriation_thrombolysis)) +
                            (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU * (1-nonLVO_known_onsettime_thrombolysis_eligible) * (repatriation_notthrombolysis)) +
                            (mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU * (repatriation_notthrombolysis)) +
                            (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU * (repatriation_notthrombolysis)))

# third array are those that go to HASU and stay
admissions_array=np.append(admissions_array,
                           (ALL_ADMISSIONS_nearestHASU -
                            LVO_ADMISSIONS_nearestHASU_bothTreatments))
# fourth array are those that go to HASU and onward to CSC [=LVO eligible for thrombectomy but went to HASU as either live too far from CSC or diagnostic test incorrectly sent them to HASU)
admissions_array = np.append(admissions_array,
                             LVO_ADMISSIONS_nearestHASU_bothTreatments)
#np.append() can only take 2 arrays, but want to append 4 so do in 3 steps:
time_array=np.append(patient_travel_matrix_MS,
                     patient_travel_matrix_to_CSC_onwards_HASU)
time_array=np.append(time_array,
                     patient_travel_matrix_All)
time_array=np.append(time_array,
                     patient_travel_matrix_to_HASU_onwards_CSC)

print("Check have all the admissions for the [total_time_travel_noDiagnosticAllocation] array: ", np.sum(admissions_array))
#Time travelled with this diagnostic test -->  total_time_travel_diag
total_time_travel_noDiagnosticAllocation=np.sum(admissions_array*time_array)#includes 60 minute turn around for any patient changing location

#Where all do go, total minutes
# first array section are those to go to CSC and stay [= those that live nearest plus any diagnostic test sends there and not repatriated],
# the second array are those to go to CSC and repatriated to HASU
# third array are those that go to HASU and stay
# fourth array are those that go to HASU and onward to CSC [=LVO eligible for thrombectomy but went to HASU as either live too far from CSC or diagnostic test incorrectly sent them to HASU)

#np.append() can only take 2 arrays, but want to append 4 so do in 3 steps:
admissions_array=np.append((ALL_ADMISSIONS_nearestCSC +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Stay +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +
                            nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +
                            nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Stay +
                            mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay +
                            haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay),
                           (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Repat +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat +
                            nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat +
                            nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Repat +
                            mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat +
                            haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat))
admissions_array=np.append(admissions_array,
                           (ALL_ADMISSIONS_nearestHASU -
                            LVO_ADMISSIONS_nearestHASU_bothTreatments +
                            ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly +
                            LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment +
                            nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU +
                            mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU +
                            haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU))
admissions_array = np.append(admissions_array,
                             LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments +
                             LVO_ADMISSIONS_nearestHASU_bothTreatments)
#np.append() can only take 2 arrays, but want to append 4 so do in 3 steps:
time_array=np.append(patient_travel_matrix_MS,
                     patient_travel_matrix_to_CSC_onwards_HASU)
time_array=np.append(time_array,
                     patient_travel_matrix_All)
time_array=np.append(time_array,
                     patient_travel_matrix_to_HASU_onwards_CSC)

print("Check have all the admissions for the [total_time_travel_diagnosticAllocation] array: ", np.sum(admissions_array))
#Time travelled with this diagnostic test -->  total_time_travel_diag
total_time_travel_diagnosticAllocation=np.sum(admissions_array*time_array)#includes 60 minute turn around for any patient changing location

#Calculate 3 arrays to calculate the admissions per centre.
# first array section are those to go to CSC and stay [= those that live nearest plus any diagnostic test sends there and not repatriated],
# second array are those that go to HASU and onward to CSC [=LVO eligible for thrombectomy but went to HASU as either live too far from CSC or diagnostic test incorrectly sent them to HASU).  CSC depends on nearest HASU and not LSOA
# thridd array are those that end up at HASU (either nearest and stay, or go to CSC and repatriated to HASU.  either way it's the same HASU, nearest to LSOA
#
#np.append() can only take 2 arrays, but want to append 4 so do in 3 steps:
CSC_admissions_array=(ALL_ADMISSIONS_nearestCSC +
                      LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments +
                      LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Stay +
                      LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +
                      nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +
                      nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Stay +
                      mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay +
                      haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay)

CSC_via_HASU_admissions_array=(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments +
                               LVO_ADMISSIONS_nearestHASU_bothTreatments)

HASU_admissions_array=(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Repat +
                       LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat +
                       nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat +
                       nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Repat +
                       mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat +
                       haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat +
                       ALL_ADMISSIONS_nearestHASU -
                       LVO_ADMISSIONS_nearestHASU_bothTreatments +
                       ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown +
                       LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly +
                       LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment +
                       nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU +
                       mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU +
                       haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU)

#Use: patient_attend_hospital_MS for the array of Patient goes to CSC and stays there

#Use: patient_attend_hospital_All for the array of Patient goes to CSC and transfer to HASU (back to nearest one to LSOA) AND Patient goes to HASU and stays there

#Use: patient_node_HASU_CSC_id [from patient_attend_hospital_All --> ID_csc_nearest_to_HASU] for the array of Patient goes to HASU and onwards to CSC (depends on HASU's closest, and not LSOA's closest)
#patient_attend_CSC_via_HASU=ID_csc_nearest_to_HASU[patient_attend_hospital_All.astype(np.int64)-1]
print("check the HASU, CSC and CSC_via_HASU admissions add up to total: ", np.sum(HASU_admissions_array)+np.sum(CSC_admissions_array)+np.sum(CSC_via_HASU_admissions_array))
print("min & max values: ", np.amin(patient_attend_hospital_All), " ", np.amax(patient_attend_hospital_All)," ", np.amin(patient_attend_hospital_MS), " ", np.amax(patient_attend_hospital_MS), " ", np.amin(patient_node_HASU_CSC_id), " ", np.amax(patient_node_HASU_CSC_id) )
HASU_admissions=np.bincount(patient_attend_hospital_All.astype(np.int64)-1, HASU_admissions_array)
CSC_admissions=np.bincount(patient_attend_hospital_MS.astype(np.int64)-1, CSC_admissions_array)
CSC_admissions_transfer=np.bincount(patient_node_HASU_CSC_id.astype(np.int64), CSC_via_HASU_admissions_array)
CSC_admissions=CSC_admissions+CSC_admissions_transfer
print("check the HASU & CSC admissions add up to total: ", np.sum(HASU_admissions)+np.sum(CSC_admissions))
print("check the shapes of the HASU admissions: ", HASU_admissions.shape, " and the CSC admissions: ", CSC_admissions.shape)

#THROMBECTOMY ADMISSIONS
CSC_thrombectomy_admissions_array=(LVO_ADMISSIONS_nearestCSC_bothTreatments +
                                   LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments)

CSC_via_HASU_thrombectomy_admissions_array=(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments +
                                            LVO_ADMISSIONS_nearestHASU_bothTreatments)

CSC_thrombectomy_admissions=np.bincount(patient_attend_hospital_MS.astype(np.int64)-1, CSC_thrombectomy_admissions_array)
CSC_thrombectomy_admissions_transfer=np.bincount(patient_node_HASU_CSC_id.astype(np.int64), CSC_via_HASU_thrombectomy_admissions_array)
CSC_thrombectomy_admissions=CSC_thrombectomy_admissions+CSC_thrombectomy_admissions_transfer

#COMBINING ADMISSIONS INTO DIFFERENT SUBGROUPS

LVO_admissions_go_to_HASU_first_CORRECT = ((LVO_ADMISSIONS_nearestHASU_noTreatment) +
                                           (LVO_ADMISSIONS_nearestHASU_tlysisOnly) +
                                           (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly) +
                                           (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment)+
                                           (ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown) * LVO)

nonLVO_admissions_go_to_HASU_first_CORRECT =((nonLVO_ADMISSIONS_nearestHASU) +
                                             (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU) +
                                             (ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown) * nonLVO)

mimic_admissions_go_to_HASU_first_CORRECT = ((mimic_ADMISSIONS_nearestHASU) +
                                             (mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU) +
                                             (ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown) * mimics)

haemorrage_admissions_go_to_HASU_first_CORRECT = ((haemorrage_ADMISSIONS_nearestHASU) +
                                                  (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU) +
                                                  (ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown) * haemorragics)

LVO_admissions_go_to_CSC_first_CORRECT = ((LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments) +
                                         (ALL_ADMISSIONS_nearestCSC) *LVO)

nonLVO_admissions_go_to_CSC_first_CORRECT = ((ALL_ADMISSIONS_nearestCSC) *nonLVO)

mimic_admissions_go_to_CSC_first_CORRECT = ((ALL_ADMISSIONS_nearestCSC) *mimics)

haemorrage_admissions_go_to_CSC_first_CORRECT = ((ALL_ADMISSIONS_nearestCSC) *haemorragics)

LVO_admissions_go_to_HASU_first_INCORRECT = ((LVO_ADMISSIONS_nearestHASU_bothTreatments) +
                                             (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments))

LVO_admissions_go_to_CSC_first_INCORRECT = ((LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly) +
                                            (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment))

nonLVO_admissions_go_to_CSC_first_INCORRECT = (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC)

mimic_admissions_go_to_CSC_first_INCORRECT = (mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC)

haemorrage_admissions_go_to_CSC_first_INCORRECT = (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC)

admissions_go_to_HASU_first_CORRECT = ((LVO_ADMISSIONS_nearestHASU_noTreatment) +
                                       (LVO_ADMISSIONS_nearestHASU_tlysisOnly) +
                                       (nonLVO_ADMISSIONS_nearestHASU) +
                                       (mimic_ADMISSIONS_nearestHASU) +
                                       (haemorrage_ADMISSIONS_nearestHASU) +
                                       (ALL_ADMISSIONS_useDiagnosticTest_onsettimeUnknown) +
                                       (mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU) +
                                       (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU) +
                                       (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU) +
                                       (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly) +
                                       (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment))

admissions_go_to_CSC_first_CORRECT = ((ALL_ADMISSIONS_nearestCSC) +
                                      (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments))

admissions_go_to_HASU_first_INCORRECT = ((LVO_ADMISSIONS_nearestHASU_bothTreatments) +
                                         (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments))

admissions_go_to_CSC_first_INCORRECT = ((mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC) +
                                        (haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC) +
                                        (nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC) +
                                        (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly) +
                                        (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment))

admissions_go_to_CSC_first=admissions_go_to_CSC_first_CORRECT+admissions_go_to_CSC_first_INCORRECT
admissions_go_to_HASU_first=admissions_go_to_HASU_first_CORRECT+admissions_go_to_HASU_first_INCORRECT
admissions_go_to_right_first =admissions_go_to_CSC_first_CORRECT+admissions_go_to_HASU_first_CORRECT
admissions_go_to_wrong_first =admissions_go_to_CSC_first_INCORRECT+admissions_go_to_HASU_first_INCORRECT

admissions_transferred_HASU_to_CSC=LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments

admissions_transferred_CSC_to_HASU=(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Repat+
                                    LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat+
                                    nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Repat+
                                    nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat+
                                    mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat+
                                    haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat)

admissions_go_to_HASU_final=admissions_go_to_HASU_first+admissions_transferred_CSC_to_HASU-admissions_transferred_HASU_to_CSC
admissions_go_to_CSC_final=admissions_go_to_CSC_first-admissions_transferred_CSC_to_HASU+admissions_transferred_HASU_to_CSC

#TIME TO THROMBOLYSIS
# 1. For the LVO patients: Time to thrombolysis (mean, median, quartiles, percentiles)
#admissions_array appends the admissions in line sutch that adding up all the admissions having the same travel time together, so that the admissions match the travel_array that had the different travel times appended.
#Fro example, the first array is the LSOA travel time to the nearest CSC, the second appended array is the LSOA travel time to the nearest HASU
admissions1_array=np.append((LVO_ADMISSIONS_nearestCSC_tlysisOnly+
                             LVO_ADMISSIONS_nearestCSC_bothTreatments+
                             LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments+
                             LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly),
                            (LVO_ADMISSIONS_nearestHASU_tlysisOnly+
                             LVO_ADMISSIONS_nearestHASU_bothTreatments+
                             LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments+
                             LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly))

time_array=np.append(patient_travel_matrix_MS,patient_travel_matrix_All)

#time to thrombolysis quartiles (min, lower, median, upper, max)
column_number=1
output_array[0,column_number]=np.sum(admissions1_array)
output_array[1:6,column_number]=wp.weighted_percentile_multiple(time_array,admissions1_array, [0,0.25,0.5,0.75,1])#... give it the values, the weights, the percentile required#time_to_thrombolysis_LVO_percentiles
#time to thrombolysis mean
time_to_thrombolysis_LVO_mean=np.sum(time_array*admissions1_array)
time_to_thrombolysis_LVO_mean=time_to_thrombolysis_LVO_mean/np.sum(admissions1_array)
output_array[6,column_number]=time_to_thrombolysis_LVO_mean

# 2. For the nonLVO patients: Time to thrombolysis (mean, median, quartiles, percentiles)
column_number=2
admissions2_array=np.append((nonLVO_ADMISSIONS_nearestCSC_tlysis+
                             nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis),
                            (nonLVO_ADMISSIONS_nearestHASU_tlysis+
                             nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysis))
#time to thrombolysis quartiles (min, lower, median, upper, max)
output_array[0,column_number]=np.sum(admissions2_array)
output_array[1:6,column_number]=wp.weighted_percentile_multiple(time_array,admissions2_array, [0,0.25,0.5,0.75,1])#... give it the values, the weights, the percentile requiredtime_to_thrombolysis_nonLVO_percentiles
#time to thrombolysis mean
time_to_thrombolysis_nonLVO_mean=np.sum(time_array*admissions2_array)
time_to_thrombolysis_nonLVO_mean=time_to_thrombolysis_nonLVO_mean/np.sum(admissions2_array)
output_array[6,column_number]=time_to_thrombolysis_nonLVO_mean

# 3. For all thrombolysis patients:  Time to thrombolysis (mean, median, quartiles, percentiles)
column_number=3
admissions_array=(admissions1_array+admissions2_array)
#time to thrombolysis quartiles (min, lower, median, upper, max)
output_array[0,column_number]=np.sum(admissions_array)
print ("Check time_array shape: ", time_array.shape)
output_array[1:6,column_number]=wp.weighted_percentile_multiple(time_array,admissions_array, [0,0.25,0.5,0.75,1])#... give it the values, the weights, the percentile required
#time to thrombolysis mean
time_to_thrombolysis_ALL_mean=np.sum(time_array*admissions_array)
time_to_thrombolysis_ALL_mean=time_to_thrombolysis_ALL_mean/np.sum(admissions_array)
output_array[6,column_number]=time_to_thrombolysis_ALL_mean

#TIME TO THROMBECTOMY
# 4. For the LVO patients: Time to thrombectomy (mean, median, quartiles, percentiles)
column_number=4
admissions_array=np.append((LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments+
                            LVO_ADMISSIONS_nearestCSC_bothTreatments),
                           (LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments+
                            LVO_ADMISSIONS_nearestHASU_bothTreatments))
time_array=np.append(patient_travel_matrix_MS,patient_travel_matrix_to_HASU_onwards_CSC)
#time to thrombectomy quartiles (min, lower, median, upper, max)
output_array[0,column_number]=np.sum(admissions_array)
output_array[1:6,column_number]=wp.weighted_percentile_multiple(time_array,admissions_array, [0,0.25,0.5,0.75,1])#... give it the values, the weights, the percentile required#time_to_thrombectomy_percentiles
#time to thrombectomy mean
time_to_thrombectomy_mean=np.sum(time_array*admissions_array)
time_to_thrombectomy_mean=time_to_thrombectomy_mean/np.sum(admissions_array)
output_array[6,column_number]=time_to_thrombectomy_mean

#time_to_thrombolysis=(np.sum(admissions_go_to_CSC_first*patient_travel_matrix_MS)+
#                      np.sum(admissions_go_to_HASU_first*patient_travel_matrix_All))

#time_to_thrombectomy=(np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments*patient_travel_matrix_to_HASU_onwards_CSC)+
#                      np.sum(LVO_ADMISSIONS_nearestHASU_bothTreatments*patient_travel_matrix_to_HASU_onwards_CSC)+
#                      np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments*patient_travel_matrix_MS)+
#                      np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments*patient_travel_matrix_MS)+
#                      np.sum(LVO_ADMISSIONS_nearestHASU_bothTreatments*patient_travel_matrix_to_HASU_onwards_CSC)+
#                      np.sum(LVO_ADMISSIONS_nearestCSC_bothTreatments*patient_travel_matrix_MS))


# 1. Proportion of LVO patients using onward transfer
proportion_thrombectomy_requiring_onwards_transfer_numerator=(np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments)+
                                                              np.sum(LVO_ADMISSIONS_nearestHASU_bothTreatments))
proportion_thrombectomy_requiring_onwards_transfer_denominator=(np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments)+
                                                                np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments)+
                                                                np.sum(LVO_ADMISSIONS_nearestHASU_bothTreatments)+
                                                                np.sum(LVO_ADMISSIONS_nearestCSC_bothTreatments))

proportion_thrombectomy_requiring_onwards_transfer=proportion_thrombectomy_requiring_onwards_transfer_numerator/proportion_thrombectomy_requiring_onwards_transfer_denominator
output_array[7,column_number]=proportion_thrombectomy_requiring_onwards_transfer

minutes_travelling_to_right_first=(np.sum(admissions_go_to_CSC_first_CORRECT*patient_travel_matrix_MS)+
                                   np.sum(admissions_go_to_HASU_first_CORRECT*patient_travel_matrix_All))

admissions_transferred_HASU_to_CSC=LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments
admissions_transferred_CSC_to_HASU=(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Repat+
                                    LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat+
                                    nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Repat+
                                    nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat+
                                    mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat+
                                    haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat)

admission_unnecessary_delay_thrombolysis=(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly+
                                          nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis)

total_unnecessary_travel_delay_thrombolysis=(np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly * patient_travel_matrix_MS)-
                                             np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly * patient_travel_matrix_All) +
                                             np.sum(nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis * patient_travel_matrix_MS) -
                                             np.sum(nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis * patient_travel_matrix_All))

column_number=5
# 2. Number of LVO admissions going to HASU first and it's correct
output_array[0,column_number]=np.sum(LVO_admissions_go_to_HASU_first_CORRECT)
# 2. Number of LVO admissions going to CSC first and it's correct
output_array[1,column_number]=np.sum(LVO_admissions_go_to_CSC_first_CORRECT)
# 2. Number of LVO admissions going to right location first
output_array[2,column_number]=np.sum(LVO_admissions_go_to_HASU_first_CORRECT+LVO_admissions_go_to_CSC_first_CORRECT)
# 3. Number of nonLVO admissions going to HASU first and it's correct
output_array[3,column_number]=np.sum(nonLVO_admissions_go_to_HASU_first_CORRECT)
# 3. Number of nonLVO admissions going to CSC first and it's correct
output_array[4,column_number]=np.sum(nonLVO_admissions_go_to_CSC_first_CORRECT)
# 3. Number of nonLVO admissions going to right location first
output_array[5,column_number]=np.sum(nonLVO_admissions_go_to_HASU_first_CORRECT+nonLVO_admissions_go_to_CSC_first_CORRECT)
# 4. Number of mimic admissions going to HASU first and it's correct
output_array[6,column_number]=np.sum(mimic_admissions_go_to_HASU_first_CORRECT)
# 4. Number of nonLVO admissions going to CSC first and it's correct
output_array[7,column_number]=np.sum(mimic_admissions_go_to_CSC_first_CORRECT)
# 4. Number of nonLVO admissions going to right location first
output_array[8,column_number]=np.sum(mimic_admissions_go_to_HASU_first_CORRECT+mimic_admissions_go_to_CSC_first_CORRECT)
# 5. Number of haemorrage admissions going to HASU first and it's correct
output_array[9,column_number]=np.sum(haemorrage_admissions_go_to_HASU_first_CORRECT)
# 5. Number of nonLVO admissions going to CSC first and it's correct
output_array[10,column_number]=np.sum(haemorrage_admissions_go_to_CSC_first_CORRECT)
# 5. Number of nonLVO admissions going to right location first
output_array[11,column_number]=np.sum(haemorrage_admissions_go_to_HASU_first_CORRECT+haemorrage_admissions_go_to_CSC_first_CORRECT)

column_number=6
# 2. Number of LVO admissions going to HASU first and it's incorrect
output_array[0,column_number]=np.sum(LVO_admissions_go_to_HASU_first_INCORRECT)
# 2. Number of LVO admissions going to CSC first and it's incorrect
output_array[1,column_number]=np.sum(LVO_admissions_go_to_CSC_first_INCORRECT)
# 2. Number of LVO admissions going to right location first
output_array[2,column_number]=np.sum(LVO_admissions_go_to_HASU_first_INCORRECT+LVO_admissions_go_to_CSC_first_INCORRECT)
# 3. Number of nonLVO admissions going to HASU first and it's incorrect
output_array[3,column_number]=0
# 3. Number of nonLVO admissions going to CSC first and it's incorrect
output_array[4,column_number]=np.sum(nonLVO_admissions_go_to_CSC_first_INCORRECT)
# 3. Number of nonLVO admissions going to right location first
output_array[5,column_number]=np.sum(nonLVO_admissions_go_to_CSC_first_INCORRECT)
# 4. Number of mimic admissions going to HASU first and it's incorrect
output_array[6,column_number]=0
# 4. Number of mimic admissions going to CSC first and it's incorrect
output_array[7,column_number]=np.sum(mimic_admissions_go_to_CSC_first_INCORRECT)
# 4. Number of mimic admissions going to right location first
output_array[8,column_number]=np.sum(mimic_admissions_go_to_CSC_first_INCORRECT)
# 5. Number of haemorrage admissions going to HASU first and it's incorrect
output_array[9,column_number]=0
# 5. Number of haemorrage admissions going to CSC first and it's incorrect
output_array[10,column_number]=np.sum(haemorrage_admissions_go_to_CSC_first_INCORRECT)
# 5. Number of haemorrage admissions going to right location first
output_array[11,column_number]=np.sum(haemorrage_admissions_go_to_CSC_first_INCORRECT)

column_number=7
# 6. Admissions going to HASU first that's correct
output_array[0,column_number]=np.sum(admissions_go_to_HASU_first_CORRECT)
# 7. Admissions going to HASU first that's incorrect
output_array[1,column_number]=np.sum(admissions_go_to_HASU_first_INCORRECT)
# 8. Admissions going to CSC first that's correct
output_array[2,column_number]=np.sum(admissions_go_to_CSC_first_CORRECT)
# 9. Admissions going to CSC first that's incorrect
output_array[3,column_number]=np.sum(admissions_go_to_CSC_first_INCORRECT)

output_array[4,column_number]=np.sum(admissions_go_to_HASU_first)
output_array[5,column_number]=np.sum(admissions_go_to_CSC_first)
output_array[6,column_number]=np.sum(admissions_go_to_right_first)
output_array[7,column_number]=np.sum(admissions_go_to_wrong_first)

output_array[8,column_number]=np.sum(admissions_transferred_CSC_to_HASU)
output_array[9,column_number]=np.sum(admissions_transferred_HASU_to_CSC)

output_array[10,column_number]=np.sum(admissions_go_to_HASU_final)
output_array[11,column_number]=np.sum(admissions_go_to_CSC_final)

#Time travelled with patients going where they need to first (effectively a 99999 decision bias and perfect diagnostic test for thrombectomy) --> total_time_travel_perfect
output_array[12,column_number]=total_time_travel_perfectAllocation#Time travelled if every patient went to correct place first
#unneccesary extra travelling (so if all patients went to their right location it would involve x mins, but some have goner to wrong location, so that's y mins.  y-x = unnecessary extra travelling.
#Time travelled with this diagnostic test -->  total_time_travel_diag
output_array[13,column_number]=total_time_travel_diagnosticAllocation
#Time travelled with a full drip n ship (0 mins decision bias) --> total_time_travel_noDiag
output_array[14,column_number]=total_time_travel_noDiagnosticAllocation#np.sum(minutes_travelling_to_right_first)#USEFUL?

output_array[15,column_number]=np.sum(admission_unnecessary_delay_thrombolysis)
output_array[16,column_number]=total_unnecessary_travel_delay_thrombolysis

# SOME PATIENTS GO TO CSC INCORRECTLY WHEN PERFECT DIAGNOSIS TEST DUE TO Those LVO patients that end up not being eligible for thrombectomy (either elig for thrombolysis and not thrombectomy OR not eligibile for thrombolysis... divided into 4 arrays to account for repatriation options)
#mimics going further to a CSC
output_array[17,column_number] =np.sum(mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC)
#mimics going further to a CSC
output_array[18,column_number] =np.sum(haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC)
#nonLVO going further to a CSC
output_array[19,column_number] = np.sum(nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC)
#nonLVO going further to a CSC (need treatrment)
output_array[20,column_number] = np.sum(nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis)
#nonLVO going further to a CSC (not need treatment)
output_array[21,column_number] = np.sum(nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment)
#LVO going unnecessarily further to CSC
output_array[22,column_number] = np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly+LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment)
#LVO only needing tlysis going further to CSC
output_array[23,column_number] = np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly)
#LVO not elig for any treatment going furhter to a CSC
output_array[24,column_number] = np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment)

#Centre Admission values (stats: min, lower qu, median, upper qu, max, mean)
column_number=8
#ENDING UP AT A HASU
print("hospital_admissions_matrix_All array shape", hospital_admissions_matrix_All.shape)
HASU_admissions=HASU_admissions[HASU_admissions != 0]
print("HASU_admissions array shape", HASU_admissions.shape)
output_array[0,column_number]=np.count_nonzero(HASU_admissions)#np.sum(population_All)#Number of HASUs
print("HASU_admissions number of centres with thrombolysis admissions: ", np.count_nonzero(HASU_admissions))
#percentiles for HASU centre admissions
output_array[1:6,column_number]=wp.weighted_percentile_multiple(HASU_admissions,None, [0,0.25,0.5,0.75,1])#... give it the values, the weights, the percentile required#time_to_thrombolysis_LVO_percentiles
#mean for HASU centre admissions
output_array[6,column_number]=np.sum(HASU_admissions)/np.count_nonzero(HASU_admissions)#np.sum(population_All)#np.count_nonzero(HASU_admissions)

#ENDING UP AT A CSC
CSC_admissions=CSC_admissions[CSC_admissions != 0]
print("CSC_admissions array shape", CSC_admissions.shape)
print("CSC_admissions array: ", CSC_admissions)
output_array[7,column_number]=np.count_nonzero(CSC_admissions)#np.sum(population_MS)#Number of CSCs
#percentiles for CSC centre admissions
output_array[8:13,column_number]=wp.weighted_percentile_multiple(CSC_admissions,None, [0,0.25,0.5,0.75,1])#... give it the values, the weights, the percentile required#time_to_thrombolysis_LVO_percentiles
#mean for CSC centre admissions
output_array[13,column_number]=np.sum(CSC_admissions)/np.count_nonzero(CSC_admissions)#np.sum(population_MS)#np.count_nonzero(CSC_admissions)

#NUMBER OF THROMBECTOMY PROCEDURES
CSC_thrombectomy_admissions=CSC_thrombectomy_admissions[CSC_thrombectomy_admissions != 0]
print("CSC_admissions array shape", CSC_thrombectomy_admissions.shape)
print("CSC_admissions array: ", CSC_thrombectomy_admissions)
output_array[14,column_number]=np.count_nonzero(CSC_thrombectomy_admissions)#np.sum(population_MS)#Number of CSCs
#percentiles for CSC centre admissions
output_array[15:20,column_number]=wp.weighted_percentile_multiple(CSC_thrombectomy_admissions,None, [0,0.25,0.5,0.75,1])#... give it the values, the weights, the percentile required#time_to_thrombolysis_LVO_percentiles
#mean for CSC centre admissions
output_array[20,column_number]=np.sum(CSC_thrombectomy_admissions)/np.count_nonzero(CSC_thrombectomy_admissions)#np.sum(population_MS)#np.count_nonzero(CSC_admissions)


#Record the inputs to the model
column_number=0
output_array[0,column_number]=LVO_as_proportion_of_ischaemic_stroke
output_array[1,column_number]=mimics_as_rate_of_stroke # rate of mimics as a % of strokes
output_array[2,column_number]=haemorragics_as_rate_of_ischaemic_strokes#Rate of haemorragics as % of ischaemic strokes
output_array[3,column_number]=onset_time_known   # Stroke symptom onset time known (used to select the patients to do the LVO diagnostic test on, else to HASU as will not be eligible for any treatment)
output_array[4,column_number]=Decision_bias# Any patient with a CSC less than this many minutes more (over a nearer HASU) will have their location determined by the diagnostic test.
output_array[5,column_number]=Specificity_stroke# Specificity for stroke patients (used on the nonLVO stroke patients)
output_array[6,column_number]=Sensitivity_stroke# Sensitivity for stroke patients (used on the LVO stroke patients)
output_array[7,column_number]=Specificity_mimics# Specificity for mimics (used on the mimics, assume less are likely to be misclassified as a LVO)
output_array[8,column_number]=nonLVO_thrombolysis_eligible # nonLVO patients that are suitable for thrombolysis
output_array[9,column_number]=nonLVO_known_onsettime_thrombolysis_eligible# LVO patients that are suitable for thrombolysis
output_array[10,column_number]=LVO_thrombolysis_eligible# LVO patients that are suitable for thrombolysis
output_array[11,column_number]=LVO_known_onsettime_thrombolysis_eligible# LVO patients that are suitable for thrombolysis
output_array[12,column_number]=LVO_thrombolysis_thrombectomy_eligible# Of those LVO patients suitable for thrombolysis, proportion suitable for thrombectomy
output_array[13,column_number]=repatriation_thrombolysis  # Proportion patients had thrombolysis that are repatriated
output_array[14,column_number]=repatriation_notthrombolysis# Proportion patients not had thrombolysis that are repatriated
output_array[15,column_number]=mimics
output_array[16,column_number]=strokes
output_array[17,column_number]=ischaemic
output_array[18,column_number]=haemorragics
output_array[19,column_number]=LVO
output_array[20,column_number]=nonLVO
output_array[15,column_number]=LVO*LVO_thrombolysis_eligible+nonLVO*nonLVO_thrombolysis_eligible#Get thrombolysis
output_array[16,column_number]=LVO*LVO_thrombolysis_eligible*LVO_thrombolysis_thrombectomy_eligible#Get thrombectomy

#ADMISSIONS
# 1. Number going to a CSC - First allocation [admissions_go_to_CSC_first]
# 2. Number going to a HASU - First allocation [admissions_go_to_HASU_first]
# 3. Number going to a CSC - Second allocation [admissions_go_to_CSC_final]
# 4. Number going to a HASU - Second allocation [admissions_go_to_HASU_final]
# 5. Per centre the admissions for first allocation (distribution statistics about centre size to know whether CSCs are too large, or HASUs are too small)
# 6. Per centre the admissions for second allocation (distribution statistics about centre size to know whether CSCs are too large, or HASUs are too small)

# check how many go to wrong & right location admissions
print(np.sum(admissions_go_to_wrong_first + admissions_go_to_right_first), " ", np.sum(ADMISSIONS))
print("Wrong: ", np.sum(admissions_go_to_wrong_first) / np.sum(ADMISSIONS), " Right: ",
      np.sum(admissions_go_to_right_first) / np.sum(ADMISSIONS))
print("sum up the wrong and rights: ",np.sum(
    admissions_go_to_HASU_first_CORRECT + admissions_go_to_HASU_first_INCORRECT + admissions_go_to_CSC_first_CORRECT + admissions_go_to_CSC_first_INCORRECT),
      " total admissions: ", np.sum(ADMISSIONS))

print("HASU correct: ", np.sum(admissions_go_to_HASU_first_CORRECT), ". HASU incorrect: ",
      np.sum(admissions_go_to_HASU_first_INCORRECT), "CSC correct: ", np.sum(admissions_go_to_CSC_first_CORRECT),
      "CSC incorrect: ", np.sum(admissions_go_to_CSC_first_INCORRECT))
# WHICH PATIENTS GO TO CSC INCORRECTLY WHEN PERFECT DIAGNOSIS TEST?: #Those LVO patients that end up not being eligible for thrombectomy (either elig for thrombolysis and not thrombectomy OR not eligibile for thrombolysis... divided into 4 arrays to account for repatriation options)
print("Admissions wrongly going to a furhter CSC.  mimics going further to a CSC: ",np.sum(mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC), " nonLVO going further to a CSC: ",
      np.sum(nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC), " LVO only needing tlysis going further to CSC: ",
      np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly), " LVO not elig for any treatment going furhter to a CSC",
      np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment))
print("Proportion of thrombectomies that needed an onwards transfer for their treatment: ",proportion_thrombectomy_requiring_onwards_transfer)
print("LVO and nonLVO patients that had their thrombolysis unnecessarily delayed: ",np.sum(admission_unnecessary_delay_thrombolysis)," From the total number of eligible LVO and nonLVOS: ",np.sum((LVO_ADMISSIONS*LVO_thrombolysis_eligible)+(nonLVO_ADMISSIONS*nonLVO_thrombolysis_eligible)))
#weighted_percentiles: =weighted_percentile(node_results[:,0],NODE_ADMISSIONS, [0.95])... give it the values, the weights, the percentile required

np.savetxt(OUTPUT_LOCATION+'/LVO_diagnostic_output.csv',output_array,delimiter=',',newline='\n')




print ("onset time known ", np.sum(ALL_ADMISSIONS_useDiagnosticTest_onsettimeKnown), " ", np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Stay +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Repat +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments+LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment + nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU + nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Stay +nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Repat +nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat +mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay +mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat +mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU +haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay+haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat+haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU))
print ("onset time known ", np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown), " ", np.sum(LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_bothTreatments +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Stay +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysisOnly_Repat +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_bothTreatments+LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_tlysisOnly +LVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU_noTreatment))
print ("onset time known ", np.sum(nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown), " ", np.sum(nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU + nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Stay +nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_tlysis_Repat +nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Stay +nonLVO_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_noTreatment_Repat))
print ("onset time known ", np.sum(mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown), " ", np.sum(mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay +mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat +mimic_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU))
print ("onset time known ", np.sum(haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown), " ", np.sum(haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Stay+haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagLVOgotoCSC_Repat+haemorrage_ADMISSIONS_useDiagnosticTest_onsettimeKnown_diagnonLVOgotoHASU))
