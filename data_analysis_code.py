# CREATING DIRECTORIES

import datetime
import os

now = datetime.datetime.now()
directory = now.strftime("%d-%m-%Y   (%H-%M-%S)")

cwd = os.getcwd()
# main directorey "name = date & time"
path = os.path.join(cwd, directory)
os. mkdir(path)

# main levels directory-------------------------------
personal_dir = path + "/personal"
os. mkdir(personal_dir)

diseases_dir = path + "/diseases"
os. mkdir(diseases_dir)

personal_diseases_dir = path + "/personal and diseases"
os. mkdir(personal_diseases_dir)

medicalTests_dir = path + "/medical Tests"
os. mkdir(medicalTests_dir)

# subdirectories-------------------------------------------------------------------------------
#**********************************************************************************************

# 1- personal subdirectories---------------------------------------------
gender_dir = personal_dir + "/gender"
os. mkdir(gender_dir)

age_dir = personal_dir + "/age"
os. mkdir(age_dir)

weight_dir = personal_dir + "/weight"
os. mkdir(weight_dir)

height_dir = personal_dir + "/height"
os. mkdir(height_dir)

education_dir = personal_dir + "/education"
os. mkdir(education_dir)

marital_dir = personal_dir + "/marital status"
os. mkdir(marital_dir)

income_dir = personal_dir + "/income"
os. mkdir(income_dir)

insurance_dir = personal_dir + "/insurance"
os. mkdir(insurance_dir)

genera_health_dir = personal_dir + "/general health"
os. mkdir(genera_health_dir)

smoker_dir = personal_dir + "/smoker"
os. mkdir(smoker_dir)

days_dir = personal_dir + "/days active"
os. mkdir(days_dir)

bmi_dir = personal_dir + "/body mass index"
os. mkdir(bmi_dir)

waist_dir = personal_dir + "/waist size"
os. mkdir(waist_dir)

drinks_dir = personal_dir + "/drinks per day"
os. mkdir(drinks_dir)

# 2- diseases subdirectories------------------------------------------------------------
asthma_dir = diseases_dir + "/asthma"
os. mkdir(asthma_dir)

chf_dir = diseases_dir + "/cognitive heart failure"
os. mkdir(chf_dir)

cad_dir = diseases_dir + "/coronaryartery disease"
os. mkdir(cad_dir)

mi_dir = diseases_dir + "/heart attack"
os. mkdir(mi_dir)

cva_dir = diseases_dir + "/stroke"
os. mkdir(cva_dir)

copd_dir = diseases_dir + "/emphysema"
os. mkdir(copd_dir)

cancer_dir = diseases_dir + "/cancer"
os. mkdir(cancer_dir)

hypertension_dir = diseases_dir + "/hypertension"
os. mkdir(hypertension_dir)

diabetes_dir = diseases_dir + "/diabetes"
os. mkdir(diabetes_dir)

pulse_dir = diseases_dir + "/heart rate"
os. mkdir(pulse_dir)

sys_bp_dir = diseases_dir + "/systolic blood presure"
os. mkdir(sys_bp_dir)

dia_bp_dir = diseases_dir + "/diastolic blood presure"
os. mkdir(dia_bp_dir)

# 3- diseases and personal subdirectories------------------------------------------------------
d_age_dir = personal_diseases_dir + "/age"
os. mkdir(d_age_dir)

d_gender_dir = personal_diseases_dir + "/gender"
os. mkdir(d_gender_dir)

d_education_dir = personal_diseases_dir + "/education"
os. mkdir(d_education_dir)

d_smoker_dir = personal_diseases_dir + "/smoker"
os. mkdir(d_smoker_dir)

d_bmi_dir = personal_diseases_dir + "/bmi"
os. mkdir(d_bmi_dir)

d_genera_health_dir = personal_diseases_dir + "/general health"
os. mkdir(d_genera_health_dir)

d_waist_dir = personal_diseases_dir + "/waist"
os. mkdir(d_waist_dir)

# 4- medical tests subdirectories--------------------------------------------------------------
cbc_dir = medicalTests_dir + "/complete blood count"
os. mkdir(cbc_dir)

lft_dir = medicalTests_dir + "/liver function test"
os. mkdir(lft_dir)

kft_dir = medicalTests_dir + "/kidney function test"
os. mkdir(kft_dir)

cmp_dir = medicalTests_dir + "/comprehensive metabolic panel"
os. mkdir(cmp_dir)

#_______________________________________________________________________
# ignoring some warnings
import warnings
warnings.filterwarnings("ignore")
#________________________________________________________________________
# importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('Patients_Dataset.csv')

#---------------------------------------------------------------------------------------------------

#									(((((DATA WRANGLING)))))


#dropping duplicated rows except for the last row.
df.drop_duplicates(keep='last', inplace=True, ignore_index=True)

# dropping repeated patient's records except for the last record.
#df.drop_duplicates(subset='id', keep='last', inplace=True, ignore_index=True)
#----------------------------------------------------------------------------------------------------
#dropping unwanted features
df.drop(['name'], axis=True, inplace=True)
#-----------------------------------------------------------------------------------------------------
#					(1- replacing null cells in categorical data with unknown and decoding encoded data)
#gender
df['gender'][df.gender.isna()] = 'unknown'

#education
df['education'][df.education.isna()] = 'unknown'

#marital
df['marital'][df.marital.isna()] = 'unknown'

# income
df['income'][df.income.isna()] = 'unknown'
df['income'][df.income == 1] = '$0 to $4,999'
df['income'][df.income == 2] = '$5k to $9,999'
df['income'][df.income == 3] = '$10k to $14,999'
df['income'][df.income == 4] = '$15k to $19,999'
df['income'][df.income == 5] = '$20k to $24,999'
df['income'][df.income == 6] = '$25k to $34,999'
df['income'][df.income == 7] = '$35k to $44,999'
df['income'][df.income == 8] = '$45k to $54,999'
df['income'][df.income == 9] = '$55k to $64,999'
df['income'][df.income == 10] = '$65k to $74,999'
df['income'][df.income == 14] = '$75k to $99,999'
df['income'][df.income == 15] = '$100k and Over'

#insurance
df['insurance'][df.insurance.isna()] = 'unknown'

#gen_health
df['gen_health'][df.gen_health.isna()] = 'unknown'

#smoker
df['smoker'][df.smoker.isna()] = 'unknown'

#days_active
df['days_active'][df.days_active == 0.0] = '0'
df['days_active'][df.days_active == 1.0] = '1'
df['days_active'][df.days_active == 2.0] = '2'
df['days_active'][df.days_active == 3.0] = '3'
df['days_active'][df.days_active == 4.0] = '4'
df['days_active'][df.days_active == 5.0] = '5'
df['days_active'][df.days_active == 6.0] = '6'
df['days_active'][df.days_active == 7.0] = '7'
df['days_active'][df.days_active.isna()] = 'unknown'
#-------------------------------------------------------------------------------------------------
#					(2- Changing categorical ordinal data to type categoricalDtype)
#gender
Gend_levels = ['female', 'male', 'unknown']
gend_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Gend_levels)
df.gender = df.gender.astype(gend_levels)

#education
Edu_levels = ['postgraduate education', 'college or equivalent', 'secondary or equivalent',
             'preparatory', 'less than preparatory', 'unknown']
edu_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Edu_levels)
df.education = df.education.astype(edu_levels)

#marital
Mari_levels = ['married', 'widowed', 'divorced', 'separated', 'never married', 'unknown']
mari_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Mari_levels)
df.marital = df.marital.astype(mari_levels)

#income
Incm_levels = ['unknown', '$0 to $4,999', '$5k to $9,999', '$10k to $14,999', '$15k to $19,999', 
               '$20k to $24,999','$25k to $34,999', '$35k to $44,999', '$45k to $54,999', 
               '$55k to $64,999', '$65k to $74,999', '$75k to $99,999', '$100k and Over']
incm_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Incm_levels)
df.income = df.income.astype(incm_levels)

#insurance
Insur_levels = ['yes', 'no', 'unknown']
insur_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Insur_levels)
df.insurance = df.insurance.astype(insur_levels)

#gen_health
Genh_levels = ['excellent', 'very good', 'good', 'fair', 'poor', 'unknown']
genh_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Genh_levels)
df.gen_health = df.gen_health.astype(genh_levels)

#smoker
Smok_levels = ['yes', 'no', 'unknown']
smok_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Smok_levels)
df.smoker = df.smoker.astype(smok_levels)

#days_active
Dact_levels = ['0', '1', '2', '3', '4', '5', '6', '7', 'unknown']
dact_levels = pd.api.types.CategoricalDtype(ordered=True, categories=Dact_levels)
df.days_active = df.days_active.astype(dact_levels)
#-------------------------------------------------------------------------------------------------------
#							(3- More wrangling: removing outliers (ranges))

# nulling all bmi cells with value greater than 110 or less than 12
df['bmi'][df.bmi > 110] = np.nan
df['bmi'][df.bmi < 12] = np.nan

# nulling all waist_cm cells with value greater than 180 or less than 30
df['waist_cm'][df.waist_cm > 180] = np.nan
df['waist_cm'][df.waist_cm < 30] = np.nan

# nulling all drinks_day cells with value greater than 100 or less than 0
df['drinks_day'][df.drinks_day > 100] = np.nan
df['drinks_day'][df.drinks_day < 0] = np.nan

# nulling all weight_kg cells with value greater than 210 or less than 40
df['weight_kg'][df.weight_kg > 250] = np.nan
df['weight_kg'][df.weight_kg < 0] = np.nan

# nulling all height_cm cells with value greater than 210 or less than 40
df['height_cm'][df.height_cm > 210] = np.nan
df['height_cm'][df.height_cm < 40] = np.nan
#-------------------------------------------------------------------------------------------------------------

#												(((PLOTTING PERSONAL DATA ANALYTICS)))

# 									{{1- single variable analysis}}
# [a- gender]
plt.figure(figsize=[16,8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='gender', palette = ['pink', 'cornflowerblue']);
sorted_counts = df.gender.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = sorted_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

    # print the annotation just above the top of the bar
    plt.text(loc, count+(sorted_counts[0]/100), pct_string, ha = 'center', color = 'black')

plt.title('Gender Proportions & Counts', fontsize= 15, pad=10)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = ['pink', 'cornflowerblue']

plt.pie(x=sorted_counts, labels=sorted_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
plt.axis('square')
plt.xlabel('gender', labelpad=30)
plt.title('Gender Proportions', fontsize= 15, pad= 25);

plt.savefig(gender_dir+"/" + "genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# [b- age]

plt.figure(figsize=[12, 7])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
ticks = np.arange(0, df.age.max()+3, 3)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df, x='age', bins= bins)
plt.xticks(ticks, labels)

plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution-histogram");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 8])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
sb.boxplot(data=df, x='age',  color=sb.color_palette('rainbow', 10)[2])

ticks = np.arange(0,120,10)
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Age Distribution', fontsize= 15)
plt.xlabel('Age', fontsize=12, labelpad=10);


#_______________________________second plot____________________________________
plt.subplot(2,1,2)
sb.violinplot(data=df, x='age', orient='horizontal', color=sb.color_palette('rainbow', 10)[2], inner= None)

plt.xticks(ticks, labels)
plt.title('', fontsize= 15)
plt.xlabel('Age', fontsize=12, labelpad=10);
plt.ylabel('', fontsize=12, labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution-box-violin");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,9])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)
sb.kdeplot(data=df, x='age', cut=0, fill= True, color="#00AFBB");
plt.title('Age Distribution Density', fontsize= 15)
plt.ylabel('Density', labelpad=10)
plt.xlabel('Age', labelpad=10);
plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

#_______________________________second plot____________________________________
plt.subplot(1,2,2)
sb.kdeplot(data=df, x='age', cumulative=True);
plt.title('Age Cumulative Distribution Function', fontsize= 15)
plt.ylabel('Propability', labelpad=10)
plt.xlabel('Age', labelpad=10);
plt.savefig(age_dir+"/" + "Age Cumulative Distribution Function");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
age_max_log = np.log10(df.age.max())
age_min_log = np.log10(df.age.min())
step = (age_max_log - age_min_log) / 40

plt.figure(figsize=[16, 8])
sb.set_theme(style="whitegrid")

bins= 10 ** np.arange(age_min_log, age_max_log + step, step)
#ticks = 10 ** np.arange(age_min_log, age_max_log + step, step * 4)
ticks = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 110]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df, x='age', bins=bins)
plt.xscale('log')
plt.xticks(ticks, labels)

plt.title('Age Distribution (Log_Transformed)', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age "Log-transformed"', labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution (Log_Transformed)");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,8])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)
sb.kdeplot(data=df, x='age', cut=0, fill= True, log_scale=True);
ticks = [1, 3, 10, 30, 60,  80, 110]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Age Distribution Density (Log_Transformed)', fontsize= 15)
plt.ylabel('Density', labelpad=10)
plt.xlabel('Age "Log-transformed"', labelpad=10);


#_______________________________second plot____________________________________
plt.subplot(1,2,2)
sb.kdeplot(data=df, x='age', cumulative=True, log_scale=True);
plt.title('Age CDF (Log_Transformed)', fontsize= 15)
plt.ylabel('Propability', labelpad=10)
plt.xlabel('Age "Log-transformed"', labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution Density (Log_Transformed)");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# [c- Education]
plt.figure(figsize=[15, 8])
sb.set_theme(style="whitegrid")

sb.countplot(data=df, x='education', palette='Blues_r')

education_counts = df.education.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = education_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

    # print the annotation just above the top of the bar
    plt.text(loc, count+(education_counts[0]/100), pct_string, ha = 'center', color = 'black')

plt.xticks(rotation=20)
plt.title("Patients' Education Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Education Level', labelpad=20);
plt.savefig(education_dir+"/" + "Patients' Education Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{d- Marital}

plt.figure(figsize=[16, 8])
sb.set_theme(style="whitegrid")

sb.countplot(data=df, x='marital', palette='BuGn_r')

marital_counts = df.marital.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = marital_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

    # print the annotation just above the top of the bar
    plt.text(loc, count+(marital_counts[0]/100), pct_string, ha = 'center', color = 'black')
    
plt.xticks(rotation=20);
plt.title("Patients' Marital Status Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Marital Status', labelpad=20);
plt.savefig(marital_dir+"/" + "Patients' Marital Status Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{e- Income}

plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

sb.countplot(data=df, x='income', palette='BuPu')

income_counts = df.income.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = income_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

    # print the annotation just above the top of the bar
    plt.text(loc, count+(income_counts[0]/100), pct_string, ha = 'center', color = 'black')
    
plt.xticks(rotation=30);
plt.title("Patients' Family Income Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Income Range', labelpad=20);
plt.savefig(income_dir+"/" + "Patients' Family Income Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{f- Insurance}

sorted_counts = df.insurance.value_counts()

#to manage plot size
plt.figure(figsize=[16,8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='insurance', order=sorted_counts.index,
             palette = ['limegreen', 'tomato', 'black']);
insurance_counts = df.insurance.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = insurance_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

    # print the annotation just below the top of the bar
    plt.text(loc, count+(sorted_counts[0]/100), pct_string, ha = 'center', color = 'black')

plt.xlabel('Insured?', labelpad=15)
plt.ylabel('Count', labelpad=10)
plt.title("Patients' Insurance Distribution", fontsize= 15, pad=15)


#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = ['limegreen', 'tomato', 'black']

plt.pie(x=sorted_counts, labels=sorted_counts.index, startangle=90, counterclock=True, 
        colors=colors, autopct=lambda p: '{:.1f}%'.format(p))
plt.axis('square')
plt.xlabel('Insured?', labelpad=30)
plt.title("Insurance Proportions", fontsize= 15, pad=50);
plt.savefig(insurance_dir+"/" + "Patients' Insurance Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{g- General Health}

plt.figure(figsize=[15, 8])
sb.set_theme(style="whitegrid")

sb.countplot(data=df, x='gen_health', palette='Greens_r')

health_counts = df.gen_health.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = health_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

    # print the annotation just above the top of the bar
    plt.text(loc, count+(health_counts[0]/100), pct_string, ha = 'center', color = 'black')
    
plt.xticks(rotation=0);
plt.title("Patients' Health State Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Health State', labelpad=20);
plt.savefig(genera_health_dir+"/" + "Patients' Health State Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{h- Smoker}

plt.figure(figsize=[15,10])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='smoker', palette = ['lightslategray', 'lawngreen', 'cornflowerblue']);
smok_counts = df.smoker.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = smok_counts[label.get_text()]
    pct_string = '{:.0f}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(smok_counts[0]/100), pct_string, ha = 'center', color = 'black')

plt.title('Smoking State Distribution', fontsize= 15)
plt.xlabel('Smoker?', labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = ['lawngreen', 'cornflowerblue', 'lightslategray']

plt.pie(x=smok_counts, labels=smok_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })

plt.axis('square')
plt.xlabel('Smoker?', labelpad=30)
plt.title('Smoking State Proportions', fontsize= 15);
plt.savefig(smoker_dir+"/" + "Patients' Health State Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{i- days_active}
plt.figure(figsize=[15, 8])
sb.set_theme(style="whitegrid")

sb.countplot(data=df, x='days_active', palette='Greens')

days_counts = df.days_active.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = days_counts[label.get_text()]
    pct_string = '{:0.1f}%'.format(100*count/df.shape[0])

    # print the annotation just above the top of the bar
    plt.text(loc, count+(days_counts[0]/100), pct_string, ha = 'center', color = 'black')
    
plt.xticks(rotation=0);
plt.title("Distribution Of Patients By Number of Active Days ", fontsize= 15, pad=10)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Number of Days per Week', labelpad=20);
plt.savefig(days_dir+"/" + "Distribution Of Patients By Number of Active Days");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{j- bmi}
plt.figure(figsize=[15, 8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.bmi.max()+.5, .5)
plt.hist(data=df, x='bmi', bins= bins)
ticks = np.arange(0, df.bmi.max()+5, 5)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(5,)
plt.axvline(x=df.bmi.mean(), linestyle='-', linewidth=3, color='yellow')

plt.title('Body-Mass-Index Distribution', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('BMI', labelpad=15);
plt.savefig(bmi_dir+"/" + "Body-Mass-Index Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{waist_cm}
plt.figure(figsize=[16, 8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.waist_cm.max()+1, 1)
ticks = np.arange(0, df.waist_cm.max()+10, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df, x='waist_cm', bins= bins)
plt.xticks(ticks, labels)
plt.xlim(30,)

plt.title('waist circumference Distribution', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Waist (cm)', labelpad=15);
plt.savefig(waist_dir+"/" + "waist circumference Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{drinks_day}
plt.figure(figsize=[20, 10])
sb.set_theme(style="darkgrid")

index_ordered = df.drinks_day.value_counts().index.sort_values()
index_ordered = index_ordered.astype('int32')

sb.countplot(data=df, y='drinks_day', order=index_ordered, color=sb.color_palette()[0])
plt.title('Distribution Of Number Of Drinks Per Day', fontsize= 15)
plt.ylabel('Drinks / Day', labelpad=20)
plt.xlabel('Count', labelpad=15);
plt.savefig(drinks_dir+"/" + "Distribution Of Number Of Drinks Per Day");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{weight_kg}
plt.figure(figsize=[20, 10])
sb.set_theme(style="darkgrid")

bins = np.arange(0, df.weight_kg.max()+2, 2)
ticks = np.arange(0, df.weight_kg.max()+10, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df, x='weight_kg', bins= bins)
plt.xticks(ticks, labels)

plt.title("Patients' Weights Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Weight (kg)', labelpad=15);
plt.savefig(weight_dir+"/" + "Patients' Weights Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{height_cm}
plt.figure(figsize=[20, 10])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.height_cm.max()+2, 2)
ticks = np.arange(40, df.height_cm.max()+10, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df, x='height_cm', bins= bins)
plt.xticks(ticks, labels)
plt.xlim(40,)

plt.title("Patients' Heights Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Height (cm)', labelpad=15);
plt.savefig(height_dir+"/" + "Patients' Heights Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#		{Bivariate exploration}
#	{gender with age}
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+1, 1)
ticks = np.arange(0, df.age.max()+5, 5)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df[df.gender == 'female'], x='age', bins= bins, color='pink')
#_______________________________first plot____________________________________

plt.hist(data=df[df.gender == 'male'], x='age', bins= bins, color='cornflowerblue', alpha=.5)
plt.legend(['female','male']);

plt.xticks(ticks, labels)

plt.title('Age Distribution For Both Genders', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);
plt.savefig(gender_dir+"/" + "Age Distribution For Both Genders-his");
plt.savefig(age_dir+"/" + "Age Distribution For Both Genders-hist");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

df['age'][df.gender == 'female'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '5', fontsize=20, 
                                                                    color='pink');
df['age'][df.gender == 'male'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '5', fontsize=20);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Age Distribution For Females and Males', fontsize= 25)
plt.xlabel('Age', fontsize= 20)
plt.ylabel('Count', fontsize= 20)
plt.legend(['Female','Male'],fontsize=20);
plt.savefig(age_dir+"/" + "Age Distribution For Both Genders-line");
plt.savefig(gender_dir+"/" + "Age Distribution For Both Genders-line");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 10])
sb.set_theme(style="whitegrid")
 
#_______________________________first plot____________________________________
plt.subplot(1,5,1)
sb.boxplot(data=df, x='gender', y='age', order=['female', 'male'], palette=['pink', 'b'], boxprops=dict(alpha=.99))
plt.title('Age For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Age', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,5,3)
sb.violinplot(data=df, x='gender', y='age', inner=None, order=['female', 'male'], palette=['pink', 'b'])
plt.title('Age For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Age', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,5,5)
sb.barplot(data=df, x='gender', y='age', errwidth=0, order=['female', 'male'], palette=['pink', 'b'])

plt.title('Average Age For\nBoth Genders', fontsize= 15, pad=15)
plt.xlabel('Gender', fontsize=12, labelpad=20)
plt.ylabel('Average Age', fontsize=12, labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution For Both Genders-box-violin-bar");
plt.savefig(gender_dir+"/" + "Age Distribution For Both Genders-line-box-violin-bar");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)
sb.kdeplot(data=df, x='age', hue='gender', hue_order=['male', 'female'], cut=0, fill= True);
plt.title('Age Distribution Density', fontsize= 15)
plt.ylabel('Density', labelpad=10)
plt.xlabel('Age', labelpad=10);

#_______________________________second plot____________________________________
plt.subplot(1,2,2)
sb.kdeplot(data=df, hue='gender', hue_order=['male', 'female'], x='age', cumulative=True);
plt.title('Age Cumulative Distribution Function', fontsize= 15)
plt.ylabel('Propability', labelpad=10)
plt.xlabel('Age', labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution For Both Genders-density");
plt.savefig(gender_dir+"/" + "Age Distribution For Both Genders-density");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gender with weight}
plt.figure(figsize=[19, 10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.weight_kg.max()+2, 2)
ticks = np.arange(0, df.weight_kg.max()+1, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df[df.gender == 'female'], x='weight_kg', bins= bins, color='pink')
#_______________________________first plot____________________________________

plt.hist(data=df[df.gender == 'male'], x='weight_kg', bins= bins, color='cornflowerblue', alpha=.5)
plt.legend(['female','male']);

plt.xticks(ticks, labels)

plt.title('Weight Distribution For Both Genders', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Weight', labelpad=10);
plt.savefig(gender_dir+"/" + "Weight Distribution For Both Genders");
plt.savefig(weight_dir+"/" + "Weight Distribution For Both Genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

df['weight_kg'][df.gender == 'female'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '5', fontsize=20, 
                                                                    color='pink');
df['weight_kg'][df.gender == 'male'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '5', fontsize=20);

ticks = np.arange(0, df.weight_kg.max()+1, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)

plt.title('Weight Distribution for Females and Males', fontsize= 25)
plt.xlabel('Weight', fontsize= 20, labelpad=20)
plt.ylabel('Count', fontsize= 20)
plt.legend(['Female','Male'],fontsize=20);
plt.savefig(gender_dir+"/" + "Weight Distribution for Females and Males");
plt.savefig(weight_dir+"/" + "Weight Distribution for Females and Males");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 8])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,3,1)
sb.boxplot(data=df, x='gender', y='weight_kg', order=['female', 'male'], palette=['pink', 'b'])
plt.title('Weight For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Weight (kg)', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,5,3)
sb.violinplot(data=df, x='gender', y='weight_kg', inner=None, order=['female', 'male'], palette=['pink', 'b'])
plt.title('Wieght For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Weight (kg)', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,3,3)
sb.barplot(data=df, x='gender', y='weight_kg', errwidth=0, order=['female', 'male'], palette=['pink', 'b'])

plt.title('Average Weight For\nBoth Genders', fontsize= 15, pad=15)
plt.xlabel('Gender', fontsize=12, labelpad=20)
plt.ylabel('Average Weight (kg)', fontsize=12, labelpad=10);
plt.savefig(gender_dir+"/" + "Weight Distribution for Both Genders-box-violin-bar");
plt.savefig(weight_dir+"/" + "Weight Distribution for Both Genders-box-violin-bar");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 	{gender with height}
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.height_cm.max()+2, 2)
ticks = np.arange(0, df.height_cm.max()+1, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df[df.gender == 'female'], x='height_cm', bins= bins, color='pink')


#_______________________________second plot____________________________________

plt.hist(data=df[df.gender == 'male'], x='height_cm', bins= bins, color='cornflowerblue', alpha=.5)
plt.legend(['female','male']);

plt.xticks(ticks, labels)
plt.xlim(45,)

plt.title('Height Distribution For Both Genders', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Height', labelpad=10);
plt.savefig(gender_dir+"/" + "Height Distribution For Both Genders");
plt.savefig(height_dir+"/" + "Height Distribution For Both Genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

df['height_cm'][df.gender == 'female'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '5', fontsize=20, 
                                                                    color='pink');
df['height_cm'][df.gender == 'male'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '5', fontsize=20);

ticks = np.arange(0, df.height_cm.max()+1, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.xlim(45,)


plt.title('Height Distribution for Females and Males', fontsize= 25)
plt.xlabel('Height', fontsize= 20, labelpad=20)
plt.ylabel('Count', fontsize= 20)
plt.legend(['Female','Male'],fontsize=20);
plt.savefig(gender_dir+"/" + "Height Distribution For Both Genders-line");
plt.savefig(height_dir+"/" + "Height Distribution For Both Genders-line");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 8])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,3,1)
sb.boxplot(data=df, x='gender', y='height_cm', order=['female', 'male'], palette=['pink', 'b'])
plt.title('Height For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Height (cm)', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,5,3)
sb.violinplot(data=df, x='gender', y='height_cm', inner=None, order=['female', 'male'], palette=['pink', 'b'])
plt.title('Height For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Height (cm)', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,3,3)
sb.barplot(data=df, x='gender', y='height_cm', errwidth=0, order=['female', 'male'], palette=['pink', 'b'])

plt.title('Average Height For\nBoth Genders', fontsize= 15, pad=15)
plt.xlabel('Gender', fontsize=12, labelpad=20)
plt.ylabel('Average Height (cm)', fontsize=12, labelpad=10);
plt.savefig(gender_dir+"/" + "Height Distribution For Both Genders-box-violin-bar");
plt.savefig(height_dir+"/" + "Height Distribution For Both Genders-box-violin-bar");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gender with education}
plt.figure(figsize=[15, 8])
sb.set_theme(style="whitegrid")

sb.countplot(data=df, x='education', hue='gender', hue_order=['female', 'male'], palette='Blues')

plt.xticks(rotation=20)
plt.title("Patients' Education Distribution For Both Genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Education Level', labelpad=20);
plt.savefig(gender_dir+"/" + "Patients' Education Distribution For Both Genders");
plt.savefig(education_dir+"/" + "Patients' Education Distribution For Both Genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gender with marital}
plt.figure(figsize=[15, 8])
sb.set_theme(style="whitegrid")

sb.countplot(data=df, x='marital', hue='gender', hue_order=['female', 'male'], palette='BuGn')
    
plt.xticks(rotation=20);
plt.title("Patients' Marital Status Distribution for Both Genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Marital Status', labelpad=20);
plt.savefig(gender_dir+"/" + "Patients' Marital Status Distribution for Both Genders");
plt.savefig(marital_dir+"/" + "Patients' Marital Status Distribution for Both Genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gender with income}
plt.figure(figsize=[18, 7])
sb.set_theme(style="whitegrid")

sb.countplot(data=df, x='income', hue='gender', hue_order=['female', 'male'], palette='BuPu')
    
plt.xticks(rotation=30);
plt.title("Patients' Family Income Distribution by Gender", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Income Range', labelpad=20);
plt.savefig(gender_dir+"/" + "Patients' Family Income Distribution by Gender");
plt.savefig(income_dir+"/" + "Patients' Family Income Distribution by Gender");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# {gender with insurance}
sorted_counts = df.insurance.value_counts()

plt.figure(figsize=[12,8])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='insurance', order=sorted_counts.index, hue='gender', hue_order=['female', 'male'],
             palette = ['peachpuff', 'blue']);

plt.xlabel('Insured?', labelpad=15)
plt.ylabel('Count', labelpad=10)
plt.title("Patients' Insurance Distribution for Both Genders", fontsize= 15, pad=15);
plt.savefig(gender_dir+"/" + "Patients' Insurance Distribution for Both Genders");
plt.savefig(insurance_dir+"/" + "Patients' Insurance Distribution for Both Genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gender with general health}
plt.figure(figsize=[16, 8])
sb.set_theme(style="whitegrid")

sb.countplot(data=df, x='gen_health', hue='gender', hue_order=['female', 'male'], palette=['peachpuff', 'blue'])
    
plt.xticks(rotation=0);
plt.title("Patients' Health State Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Health State', labelpad=20);
plt.savefig(gender_dir+"/" + "Patients' Health State Distribution for both genders");
plt.savefig(genera_health_dir+"/" + "Patients' Health State Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 	{gender with smoke}
plt.figure(figsize=[16,8])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='smoker', hue='gender', hue_order=['female', 'male'], palette=['peachpuff', 'blue']);

plt.title('Smoking State Distribution by Gender', fontsize= 15)
plt.xlabel('Smoker?', labelpad=20);
plt.savefig(gender_dir+"/" + "Smoking State Distribution by Gender");
plt.savefig(smoker_dir+"/" + "Smoking State Distribution by Gender");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{BMI & gender}
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.bmi.max()+.5, .5)
ticks = np.arange(0, df.bmi.max()+5, 5)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df[df.gender == 'female'], x='bmi', bins= bins, color='pink')
#_______________________________first plot____________________________________

plt.hist(data=df[df.gender == 'male'], x='bmi', bins= bins, color='cornflowerblue', alpha=.5)
plt.legend(['female','male']);

plt.xticks(ticks, labels)
plt.xlim(5,)

plt.title('BMI Distribution For Both Genders', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('BMI', labelpad=10);
plt.savefig(gender_dir+"/" + "BMI Distribution For Both Genders");
plt.savefig(bmi_dir+"/" + "BMI Distribution For Both Genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

df['bmi'][df.gender == 'female'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='pink');
df['bmi'][df.gender == 'male'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20, alpha =.5);

ticks = np.arange(0, df.bmi.max()+5, 5)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('bmi Distribution For Females and Males', fontsize= 25)
plt.xlabel('bmi', fontsize= 20)
plt.ylabel('Count', fontsize= 20)
plt.legend(['Female','Male'],fontsize=20);
plt.savefig(gender_dir+"/" + "BMI Distribution For Both Genders-line");
plt.savefig(bmi_dir+"/" + "BMI Distribution For Both Genders-line");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,3,1)
sb.boxplot(data=df, x='gender', y='bmi', order=['female', 'male'], palette=['pink', 'b'])
plt.title('BMI For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('BMI (kg/m^2)', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,5,3)
sb.violinplot(data=df, x='gender', y='bmi', inner=None, order=['female', 'male'], palette=['pink', 'b'])
plt.title('BMI For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('BMI (kg/m^2)', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,3,3)
sb.barplot(data=df, x='gender', y='bmi', errwidth=0, order=['female', 'male'], palette=['pink', 'b'])

plt.title('Average BMI For\nBoth Genders', fontsize= 15, pad=15)
plt.xlabel('Gender', fontsize=12, labelpad=20)
plt.ylabel('Average BMI (kg/m^2)', fontsize=12, labelpad=10);
plt.savefig(gender_dir+"/" + "BMI Distribution For Both Genders-box-violin-bar");
plt.savefig(bmi_dir+"/" + "BMI Distribution For Both Genders-box-violin-bar");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gender with waist}
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.waist_cm.max()+1, 1)
ticks = np.arange(0, df.waist_cm.max()+10, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df[df.gender == 'female'], x='waist_cm', bins= bins, color='pink')
#_______________________________second plot____________________________________

plt.hist(data=df[df.gender == 'male'], x='waist_cm', bins= bins, color='cornflowerblue', alpha=.5)
plt.legend(['female','male']);

plt.xticks(ticks, labels)
plt.xlim(30,)

plt.title('Waist Distribution For Both Genders', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Waist (cm)', labelpad=10);
plt.savefig(gender_dir+"/" + "Waist Distribution For Both Genders");
plt.savefig(waist_dir+"/" + "Waist Distribution For Both Genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 10])

df['waist_cm'][df.gender == 'female'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='pink');
df['waist_cm'][df.gender == 'male'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20, alpha =.5);

ticks = np.arange(0, df.waist_cm.max()+10, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)


plt.title('Waist Size Distribution For Females and Males', fontsize= 25)
plt.xlabel('Waist (cm)', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 20, labelpad=10)
plt.legend(['Female','Male'],fontsize=20);
plt.savefig(gender_dir+"/" + "Waist Distribution For Both Genders-line");
plt.savefig(waist_dir+"/" + "Waist Distribution For Both Genders-line");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,3,1)
sb.boxplot(data=df, x='gender', y='waist_cm', order=['female', 'male'], palette=['pink', 'b'])
plt.title('Waist Size For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Waist (cm)', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,3,2)
sb.violinplot(data=df, x='gender', y='waist_cm', inner=None, order=['female', 'male'], palette=['pink', 'b'])
plt.title('Waist size For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Waist (cm)', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,3,3)
sb.barplot(data=df, x='gender', y='waist_cm', errwidth=0, order=['female', 'male'], palette=['pink', 'b'])

plt.title('Average Waist size For\nBoth Genders', fontsize= 15, pad=15)
plt.xlabel('Gender', fontsize=12, labelpad=20)
plt.ylabel('Average Waist (cm)', fontsize=12, labelpad=10);
plt.savefig(gender_dir+"/" + "Waist vs Genders-box-violin-bar");
plt.savefig(waist_dir+"/" + "Waist vs Genders-box-violin-bar");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gender with drinks_day}
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
bins = np.arange(0, df.drinks_day.max()+1, 1)
ticks = np.arange(0, df.drinks_day.max()+10, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.hist(data=df[df.gender == 'female'], x='drinks_day', bins= bins, color='pink')
#_______________________________second plot____________________________________

plt.hist(data=df[df.gender == 'male'], x='drinks_day', bins= bins, color='cornflowerblue', alpha=.5)
plt.legend(['female','male']);

plt.xticks(ticks, labels)


plt.title('Drinks/Day Distribution For Both Genders', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Drinks/Day', labelpad=10);
plt.savefig(gender_dir+"/" + "Drinks per Day Distribution For Both Genders");
plt.savefig(drinks_dir+"/" + "Drinks per Day Distribution For Both Genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

df['drinks_day'][df.gender == 'female'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='pink');
df['drinks_day'][df.gender == 'male'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20, alpha =.5);

ticks = np.arange(0, df.drinks_day.max()+10, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)


plt.title('Drinks/day Distribution For Females and Males', fontsize= 25)
plt.xlabel('Drinks/day', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 20, labelpad=10)
plt.legend(['Female','Male'],fontsize=20);
plt.savefig(gender_dir+"/" + "Drinks per Day Distribution For Both Genders-line");
plt.savefig(drinks_dir+"/" + "Drinks per Day Distribution For Both Genders-line");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

#_______________________________first plot____________________________________
plt.subplot(1,3,1)
sb.boxplot(data=df, x='gender', y='drinks_day', order=['female', 'male'], palette=['pink', 'b'])
plt.title('Drinks/day For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Drinks/day', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,3,2)
sb.violinplot(data=df, x='gender', y='drinks_day', inner=None, order=['female', 'male'], palette=['pink', 'b'])
plt.title('Drinks/day For\nBoth Genders', fontsize= 15, pad=15)
plt.ylabel('Drinks/day', fontsize=12, labelpad=10);
plt.xlabel('Gender', fontsize=12, labelpad=20)

#_______________________________second plot____________________________________
plt.subplot(1,3,3)
sb.barplot(data=df, x='gender', y='drinks_day', errwidth=0, order=['female', 'male'], palette=['pink', 'b'])

plt.title('Average Drinks/day For\nBoth Genders', fontsize= 15, pad=15)
plt.xlabel('Gender', fontsize=12, labelpad=20)
plt.ylabel('Average Drinks/day', fontsize=12, labelpad=10);
plt.savefig(gender_dir+"/" + "Drinks per Day Distribution For Both Genders-box-violin-bar");
plt.savefig(drinks_dir+"/" + "Drinks per Day Distribution For Both Genders-box-violin-bar");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with weight}
plt.figure(figsize=[20, 10])
sb.set_theme(style=None)

x_bins = np.arange(0, 120, 5)
y_bins = np.arange(0, 260, 10)

heat_map = plt.hist2d(data=df, x='age', y='weight_kg', cmin=0.5, cmap='viridis_r', bins=[x_bins, y_bins])
plt.colorbar()

counts = heat_map[0]

for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        
        c = counts[i,j]
        if c>= 100 :
            plt.text(x_bins[i]+2, y_bins[j]+5, int(c), ha='center', va='center', color='white');
            
        elif c>0 :
            plt.text(x_bins[i]+2, y_bins[j]+5, int(c), ha='center', va='center', color='black');
            

x_bins = np.arange(0, 120, 10)
labels_x = ['{:.0f}'.format(v) for v in x_bins]
y_bins = np.arange(0, 260, 20)
labels_y = ['{:.0f}'.format(v) for v in y_bins]
plt.xticks(x_bins, labels_x)
plt.yticks(y_bins, labels_y)

plt.title('Weight vs Age', fontsize= 22, pad=15)
plt.xlabel('Age', fontsize= 15, labelpad=15)
plt.ylabel('Weight', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "Weight vs Age-heatmap");
plt.savefig(weight_dir+"/" + "Weight vs Age-heatmap");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with height}
plt.figure(figsize=[20, 10])
sb.set_theme(style=None)

x_bins = np.arange(0, 120, 5)
y_bins = np.arange(0, 210, 10)

heat_map = plt.hist2d(data=df, x='age', y='height_cm', cmin=0.5, cmap='viridis_r', bins=[x_bins, y_bins])
plt.colorbar()

counts = heat_map[0]

for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        
        c = counts[i,j]
        if c>= 100 :
            plt.text(x_bins[i]+2, y_bins[j]+5, int(c), ha='center', va='center', color='white');
            
        elif c>0 :
            plt.text(x_bins[i]+2, y_bins[j]+5, int(c), ha='center', va='center', color='black');
            

x_bins = np.arange(0, 120, 10)
labels_x = ['{:.0f}'.format(v) for v in x_bins]
y_bins = np.arange(0, 260, 20)
labels_y = ['{:.0f}'.format(v) for v in y_bins]
plt.xticks(x_bins, labels_x)
plt.yticks(y_bins, labels_y)

plt.ylim(45,)

plt.title('Height vs Age', fontsize= 22, pad=15)
plt.xlabel('Age', fontsize= 15, labelpad=15)
plt.ylabel('Height', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "Height vs Age-heatmap");
plt.savefig(height_dir+"/" + "Height vs Age-heatmap");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with education}
plt.figure(figsize=[25,10])
sb.set_theme(style="whitegrid")

plt.subplot(1,2,1)
sb.stripplot(x='education', y='age', data=df, jitter=0.3, color=sb.color_palette('viridis_r', 10)[0], alpha=.5)

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=20)

plt.title('Age vs Education Level', fontsize= 20, pad=15)
plt.xlabel('Education Level', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=5);

#_______________________________second plot____________________________________
plt.subplot(1,2,2)
sb.violinplot(data=df, x='education', y='age', color=sb.color_palette('viridis_r', 10)[0]) #color=sb.color_palette('viridis_r', 10)[0]

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=20)

plt.title('Age Distribution vs Education Level', fontsize= 20, pad=10)
plt.xlabel('Education Level', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution vs Education Level- scatter-violin");
plt.savefig(education_dir+"/" + "Age Distribution vs Education Level- scatter-violin");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with marital}
plt.figure(figsize=[25,10])
sb.set_theme(style="whitegrid")

plt.subplot(1,2,1)
sb.stripplot(x='marital', y='age', data=df, jitter=0.3, color=sb.color_palette('viridis_r', 10)[1], alpha=.5)

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=20)

plt.title('Age vs Marital Status', fontsize= 20, pad=15)
plt.xlabel('Marital Status', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);


plt.subplot(1,2,2)
sb.violinplot(data=df, x='marital', y='age', color=sb.color_palette('viridis_r', 10)[1]) #color=sb.color_palette('viridis_r', 10)[0]

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=20)

plt.title('Age Distribution vs Marital Status', fontsize= 20, pad=15)
plt.xlabel('Marital Status', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution vs Marital Status- scatter-violin");
plt.savefig(marital_dir+"/" + "Age Distribution vs Marital Status- scatter-violin");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with income}
plt.figure(figsize=[25,10])
sb.set_theme(style="whitegrid")

plt.subplot(1,2,1)
sb.stripplot(x='income', y='age', data=df, jitter=0.3, color=sb.color_palette('viridis_r', 10)[2], alpha=.5)

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=30)

plt.title('Age vs Income Range', fontsize= 20, pad=15)
plt.xlabel('Income Range', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);


plt.subplot(1,2,2)
sb.violinplot(data=df, x='income', y='age', color=sb.color_palette('viridis_r', 10)[2]) #color=sb.color_palette('viridis_r', 10)[0]

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=30)

plt.title('Age Distribution vs Income Range', fontsize= 20, pad=15)
plt.xlabel('Income Range', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution vs Income Range- scatter-violin");
plt.savefig(income_dir+"/" + "Age Distribution vs Income Range- scatter-violin");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with insurance}
plt.figure(figsize=[20,10])
sb.set_theme(style="whitegrid")

plt.subplot(1,2,1)
sb.stripplot(x='insurance', y='age', data=df, jitter=0.3, color=sb.color_palette('viridis_r', 10)[3], alpha=.5)

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=0)

plt.title('Age vs Insurance', fontsize= 20, pad=15)
plt.xlabel('Insurance', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);


plt.subplot(1,2,2)
sb.violinplot(data=df, x='insurance', y='age', color=sb.color_palette('viridis_r', 10)[3]) #color=sb.color_palette('viridis_r', 10)[0]

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=0)

plt.title('Age Distribution vs Insurance', fontsize= 20, pad=15)
plt.xlabel('Insurance', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution vs Insurance- scatter-violin");
plt.savefig(insurance_dir+"/" + "Age Distribution vs Insurance- scatter-violin");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with general health}
plt.figure(figsize=[25,10])
sb.set_theme(style="whitegrid")

plt.subplot(1,2,1)
sb.stripplot(x='gen_health', y='age', data=df, jitter=0.3, color=sb.color_palette('viridis_r', 10)[3], alpha=.5)

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)

plt.title('Age vs General Health Condition', fontsize= 20, pad=15)
plt.xlabel('Health Condition', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);


plt.subplot(1,2,2)
sb.violinplot(data=df, x='gen_health', y='age', color=sb.color_palette('viridis_r', 10)[3]) 
ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)

plt.title('Age Distribution vs General Health Condition', fontsize= 20, pad=15)
plt.xlabel('Health Condition', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution vs General Health Condition- scatter-violin");
plt.savefig(genera_health_dir+"/" + "Age Distribution vs General Health Condition- scatter-violin");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with smoker}
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

plt.subplot(1,2,1)
sb.stripplot(x='smoker', y='age', data=df, jitter=0.3, color=sb.color_palette('viridis_r', 10)[4], alpha=.5)

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=0)

plt.title('Age vs Smoking State', fontsize= 20, pad=15)
plt.xlabel('Smoking State', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);


plt.subplot(1,2,2)
sb.violinplot(data=df, x='smoker', y='age', color=sb.color_palette('viridis_r', 10)[4])

plt.title('Age Distribution vs Smoking State', fontsize= 20, pad=15)
plt.xlabel('Smoking State', fontsize= 15, labelpad=15)
plt.ylabel('Age', fontsize= 15, labelpad=15);
plt.savefig(age_dir+"/" + "Age Distribution vs Smoking State- scatter-violin");
plt.savefig(smoker_dir+"/" + "Age Distribution vs Smoking State- scatter-violin");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with days active}
plt.figure(figsize=[30, 15])
sb.set_theme(style="whitegrid")

plt.subplot(1,2,1)
sb.stripplot(x='days_active', y='age', data=df, jitter=0.3, color=sb.color_palette('RdBu', 10)[1], alpha=.5)

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=0)

plt.title('Age vs Number Of Days Active', fontsize= 20, pad=15)
plt.xlabel('Number Of Days Active', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);


plt.subplot(1,2,2)
sb.violinplot(data=df, x='days_active', y='age', color=sb.color_palette('RdBu', 10)[1])

ticks = np.arange(0, 120, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.xticks(rotation=0)

plt.title('Age Distribution vs Number Of Days Active', fontsize= 20, pad=15)
plt.xlabel('Number Of Days Active', fontsize= 15, labelpad=20)
plt.ylabel('Age', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "Age Distribution vs Number Of Days Active- scatter-violin");
plt.savefig(days_dir+"/" + "Age Distribution vs Number Of Days Active- scatter-violin");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with BMI}
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

x_bins = np.arange(0, 120, 5)
y_bins = np.arange(10, 110, 5)

heat_map = plt.hist2d(data=df, x='age', y='bmi', cmin=0.5, cmap='viridis_r', bins=[x_bins, y_bins])
plt.colorbar()

counts = heat_map[0]

for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        
        c = counts[i,j]
        if c>= 100 :
            plt.text(x_bins[i]+2, y_bins[j]+2, int(c), ha='center', va='center', color='white');
            
        elif c>0 :
            plt.text(x_bins[i]+2, y_bins[j]+2, int(c), ha='center', va='center', color='black');
            

x_bins = np.arange(0, 120, 10)
labels_x = ['{:.0f}'.format(v) for v in x_bins]
y_bins = np.arange(10, df.bmi.max()+5, 10)
labels_y = ['{:.0f}'.format(v) for v in y_bins]
plt.xticks(x_bins, labels_x)
plt.yticks(y_bins, labels_y)
plt.ylim(10, df.bmi.max()+1)

plt.title('BMI vs Age', fontsize= 22, pad=15)
plt.xlabel('Age', fontsize= 15, labelpad=15)
plt.ylabel('BMI', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "BMI vs Age- heat_map");
plt.savefig(bmi_dir+"/" + "BMI vs Age- heat_map");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with Waist circumference}
plt.figure(figsize=[20, 10])
sb.set_theme(style="whitegrid")

x_bins = np.arange(0, 120, 5)
y_bins = np.arange(0, 180, 10)

heat_map = plt.hist2d(data=df, x='age', y='waist_cm', cmin=0.5, cmap='viridis_r', bins=[x_bins, y_bins])
plt.colorbar()

counts = heat_map[0]

for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        
        c = counts[i,j]
        if c>= 100 :
            plt.text(x_bins[i]+2, y_bins[j]+2, int(c), ha='center', va='center', color='white');
            
        elif c>0 :
            plt.text(x_bins[i]+2, y_bins[j]+2, int(c), ha='center', va='center', color='black');
            

x_bins = np.arange(0, 120, 10)
labels_x = ['{:.0f}'.format(v) for v in x_bins]
y_bins = np.arange(10, df.waist_cm.max()+5, 10)
labels_y = ['{:.0f}'.format(v) for v in y_bins]
plt.xticks(x_bins, labels_x)
plt.yticks(y_bins, labels_y)
plt.ylim(10, df.waist_cm.max()+1)

plt.title('BMI vs Age', fontsize= 22, pad=15)
plt.xlabel('Age', fontsize= 15, labelpad=15)
plt.ylabel('Waist size', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "waist vs Age- heat_map");
plt.savefig(waist_dir+"/" + "waist vs Age- heat_map");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age with drinks day}
plt.figure(figsize=[20, 10])
#sb.set_theme(style='whitegrid')

x_bins = np.arange(0, 110, 5)
y_bins = np.arange(0, 100, 5)

heat_map = plt.hist2d(data=df, x='age', y='drinks_day', cmin=0.5, cmap='viridis_r', bins=[x_bins, y_bins])
plt.colorbar()

counts = heat_map[0]

for i in range(counts.shape[0]):
    for j in range(counts.shape[1]):
        
        c = counts[i,j]
        if c>= 100 :
            plt.text(x_bins[i]+2, y_bins[j]+2, int(c), ha='center', va='center', color='white');
            
        elif c>0 :
            plt.text(x_bins[i]+2, y_bins[j]+2, int(c), ha='center', va='center', color='black');
            

x_bins = np.arange(0, 120, 10)
labels_x = ['{:.0f}'.format(v) for v in x_bins]
y_bins = [0,5,10,15,20,25,30,40,50,60,70,80,90]
labels_y = ['{:.0f}'.format(v) for v in y_bins]
plt.xticks(x_bins, labels_x)
plt.yticks(y_bins, labels_y)

plt.title('Drinks/day vs Age', fontsize= 22, pad=15)
plt.xlabel('Age', fontsize= 15, labelpad=15)
plt.ylabel('Drinks/day', fontsize= 15, labelpad=10);
plt.savefig(age_dir+"/" + "drinks per day vs Age- heat_map");
plt.savefig(drinks_dir+"/" + "drinks per day vs Age- heat_map");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{General relations between personal numeric features}
sb.set_theme(style=None)

personal_features = ['age', 'weight_kg', 'height_cm', 'bmi','waist_cm', 'drinks_day']
g = sb.PairGrid(data=df, vars=personal_features)
g.map_offdiag(plt.scatter, alpha=.01)
g.map_diag(plt.hist);
plt.savefig(age_dir+"/" + "pair plot between age-weight-height-bmi-waist-drinks");
plt.savefig(weight_dir+"/" + "pair plot between age-weight-height-bmi-waist-drinks");
plt.savefig(height_dir+"/" + "pair plot between age-weight-height-bmi-waist-drinks");
plt.savefig(bmi_dir+"/" + "pair plot between age-weight-height-bmi-waist-drinks");
plt.savefig(waist_dir+"/" + "pair plot between age-weight-height-bmi-waist-drinks");
plt.savefig(drinks_dir+"/" + "pair plot between age-weight-height-bmi-waist-drinks");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize= [16,10])

personal_features = ['age', 'weight_kg', 'height_cm', 'bmi','waist_cm', 'drinks_day']
correlations = df[personal_features].corr()

sb.heatmap(correlations, cmap='vlag_r', annot=True, fmt='.2f', center = 0)

plt.title('Personal Data Correlations  (Paired)', fontsize= 22, pad=15)
plt.xlabel('Features', fontsize= 15, labelpad=20)
plt.ylabel('Features', fontsize= 15, labelpad=20);
plt.savefig(weight_dir+"/" + "Correlations between age-weight-height-bmi-waist-drinks");
plt.savefig(height_dir+"/" + "Correlations between age-weight-height-bmi-waist-drinks");
plt.savefig(bmi_dir+"/" + "Correlations between age-weight-height-bmi-waist-drinks");
plt.savefig(waist_dir+"/" + "Correlations between age-weight-height-bmi-waist-drinks");
plt.savefig(drinks_dir+"/" + "Correlations between age-weight-height-bmi-waist-drinks");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#				{Multivariate analysis}
plt.figure(figsize= [16, 20])

personal_features = ['age', 'weight_kg', 'height_cm', 'bmi','waist_cm', 'drinks_day']

plt.subplot(2,1,1)
correlations = df[df.gender == 'female'][personal_features].corr()
sb.heatmap(correlations, cmap='Spectral', annot=True, fmt='.2f', center = 0)

plt.title('Females Personal Data Correlations  (Paired)', fontsize= 22, pad=15)
plt.xlabel('Features', fontsize= 15, labelpad=20)
plt.ylabel('Features', fontsize= 15, labelpad=20);

plt.subplot(2,1,2)
correlations = df[df.gender == 'male'][personal_features].corr()
sb.heatmap(correlations, cmap='coolwarm', annot=True, fmt='.2f', center = 0)

plt.title('Males Personal Data Correlations  (Paired)', fontsize= 22, pad=15)
plt.xlabel('Features', fontsize= 15, labelpad=20)
plt.ylabel('Features', fontsize= 15, labelpad=20);
plt.savefig(weight_dir+"/" + "Correlations by gender between age-weight-height-bmi-waist-drinks");
plt.savefig(height_dir+"/" + "Correlations by gender between age-weight-height-bmi-waist-drinks");
plt.savefig(bmi_dir+"/" + "Correlations by gender between age-weight-height-bmi-waist-drinks");
plt.savefig(waist_dir+"/" + "Correlations by gender between age-weight-height-bmi-waist-drinks");
plt.savefig(drinks_dir+"/" + "Correlations by gender between age-weight-height-bmi-waist-drinks");
plt.savefig(gender_dir+"/" + "Correlations by gender between age-weight-height-bmi-waist-drinks");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gender, education, gen_health}
plt.figure(figsize=[25, 20])
sb.set_theme(style='whitegrid')

plt.subplot(2,3,1)
sb.countplot(data=df[(df.education == 'postgraduate education')], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='Greens_r')

plt.xticks(rotation=0)
plt.title("postgraduate education Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(2,3,2)
sb.countplot(data=df[df.education == 'college or equivalent'], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='Greens_r')

plt.xticks(rotation=0)
plt.title("'college or equivalent' Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(2,3,3)
sb.countplot(data=df[df.education == 'secondary or equivalent'], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='Greens_r')

plt.xticks(rotation=0)
plt.title("'secondary or equivalent' Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(2,3,4)
sb.countplot(data=df[df.education == 'preparatory'], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='Greens_r')

plt.xticks(rotation=0)
plt.title("'preparatory' Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(2,3,5)
sb.countplot(data=df[df.education == 'less than preparatory'], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='Greens_r')

plt.xticks(rotation=0)
plt.title("'less than preparatory' education Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(2,3,6)
sb.countplot(data=df[df.education == 'unknown'], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='Greens_r')

plt.xticks(rotation=0)
plt.title("'unknown' education Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);
plt.savefig(gender_dir+"/" + "general health vs education level for both genders");
plt.savefig(education_dir+"/" + "general health vs education level for both genders");
plt.savefig(genera_health_dir+"/" + "general health vs education level for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gender, gen_health, age}
plt.figure(figsize=[20,10])

sb.barplot(data=df, x='gender', order=['female', 'male'], y='age', hue='gen_health', palette='viridis', alpha=1)

plt.title('Average Age of Both Genders By Different Health Conditions', fontsize= 15)
plt.ylabel('Average Age', labelpad=10)
plt.xlabel('Gender', labelpad=10);
plt.savefig(gender_dir+"/" + "average age vs general health for both genders");
plt.savefig(age_dir+"/" + "average age vs general health for both genders");
plt.savefig(genera_health_dir+"/" + "average age vs general health for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{gen_health, age, bmi}
g = sb.FacetGrid(data=df, col='gen_health', 
                 col_order=['excellent', 'very good', 'good', 'fair', 'poor', 'unknown'], 
                 col_wrap=3, height=6, aspect=1.5)
g.map(sb.regplot, 'age', 'bmi', scatter_kws={'alpha':.25}, truncate=False);
plt.savefig(bmi_dir+"/" + "bmi vs age vs general health");
plt.savefig(age_dir+"/" + "bmi vs age vs general health");
plt.savefig(genera_health_dir+"/" + "bmi vs age vs general health");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{age, bmi, days_active}
plt.figure(figsize=[20, 10])
plt.scatter(data=df, x='age', y='bmi', c='waist_cm', cmap='PiYG_r', alpha=1)
plt.colorbar(label='Waist (cm)');

plt.title('BMI vs Age vs Waist circumference', fontsize= 15)
plt.ylabel('Body Mass Index', labelpad=10)
plt.xlabel('Age', labelpad=10);
plt.savefig(bmi_dir+"/" + "bmi vs age vs waist size");
plt.savefig(age_dir+"/" + "bmi vs age vs waist size");
plt.savefig(waist_dir+"/" + "bmi vs age vs waist size");
#$$$$$$$$$$$$$$$$$$
#	{gender, education, gen_health, smoker}
plt.figure(figsize=[20, 38])

sb.set_theme(style='whitegrid')

plt.subplot(4,3,1)
sb.countplot(data=df[(df.education == 'postgraduate education') & (df.smoker == 'yes')], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='Greys_r')

plt.xticks(rotation=0)
plt.title("Smoking = yes \npostgraduate education Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,2)
sb.countplot(data=df[(df.education == 'college or equivalent') & (df.smoker == 'yes')], x='gender',
            order=['female', 'male'], hue='gen_health', palette='Greys_r')

plt.xticks(rotation=0)
plt.title("Smoking = yes \n'college or equivalent' Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,3)
sb.countplot(data=df[(df.education == 'secondary or equivalent') & (df.smoker == 'yes')], x='gender',
            order=['female', 'male'], hue='gen_health', palette='Greys_r')

plt.xticks(rotation=0)
plt.title("Smoking = yes \n'secondary or equivalent' Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,4)
sb.countplot(data=df[(df.education == 'preparatory') & (df.smoker == 'yes')], x='gender',
            order=['female', 'male'], hue='gen_health', palette='Greys_r')

plt.xticks(rotation=0)
plt.title("Smoking = yes \n'preparatory' Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,5)
sb.countplot(data=df[(df.education == 'less than preparatory') & (df.smoker == 'yes')], x='gender',
            order=['female', 'male'], hue='gen_health', palette='Greys_r')

plt.xticks(rotation=0)
plt.title("Smoking = yes \n'less than preparatory' education Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,6)
sb.countplot(data=df[(df.education == 'unknown') & (df.smoker == 'yes')], x='gender',
            order=['female', 'male'], hue='gen_health', palette='Greys_r')

plt.xticks(rotation=0)
plt.title("Smoking = yes \n'unknown' education Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);
                     

#--------------------------------------------------------------------------------------
                     #----------------------------------------
                     

sb.set_theme(style='whitegrid')

plt.subplot(4,3,7)
sb.countplot(data=df[(df.education == 'postgraduate education') & (df.smoker == 'no')], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='GnBu_r')

plt.xticks(rotation=0)
plt.title("Smoking = no \npostgraduate education Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,8)
sb.countplot(data=df[(df.education == 'college or equivalent') & (df.smoker == 'no')], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='GnBu_r')

plt.xticks(rotation=0)
plt.title("Smoking = no \n'college or equivalent' Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,9)
sb.countplot(data=df[(df.education == 'secondary or equivalent') & (df.smoker == 'no')], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='GnBu_r')

plt.xticks(rotation=0)
plt.title("Smoking = no \n'secondary or equivalent' Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,10)
sb.countplot(data=df[(df.education == 'preparatory') & (df.smoker == 'no')], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='GnBu_r')

plt.xticks(rotation=0)
plt.title("Smoking = no \n'preparatory' Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,11)
sb.countplot(data=df[(df.education == 'less than preparatory') & (df.smoker == 'no')], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='GnBu_r')

plt.xticks(rotation=0)
plt.title("Smoking = no \n'less than preparatory' education Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);

# --------------------------------------------------
plt.subplot(4,3,12)
sb.countplot(data=df[(df.education == 'unknown') & (df.smoker == 'no')], x='gender', order=['female', 'male'], 
             hue='gen_health', palette='GnBu_r')

plt.xticks(rotation=0)
plt.title("Smoking = no \n'unknown' education Patients Distribution", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Gender', labelpad=10);
plt.savefig(gender_dir+"/" + "general health vs education level by gender vs smoking condition");
plt.savefig(education_dir+"/" + "general health vs education level by gender vs smoking condition");
plt.savefig(genera_health_dir+"/" + "general health vs education level by gender vs smoking condition");
plt.savefig(smoker_dir+"/" + "general health vs education level by gender vs smoking condition");
#$$$$$$$$$$$$$$$$$$$$$$$
#	{gender, education, gen_health, age}
gender_order = ['female', 'male']
edu_order = ['postgraduate education', 'college or equivalent', 'secondary or equivalent',
             'preparatory', 'less than preparatory']

health_order = ['excellent', 'very good', 'good', 'fair', 'poor']

plt.figure(figsize=[20,30])

plt.subplot(3,2,1)
sb.pointplot(data=df[df.education == 'postgraduate education'], x='gender', y='bmi', hue='gen_health', linestyles="", ci='sd', 
             dodge=.3, order=gender_order, hue_order=health_order, palette='inferno', errwidth=1.5);

plt.xticks(rotation=0)
plt.title("Education = postgraduate education \nAverage BMI vs General Health for both Genders", fontsize= 15)
plt.ylabel('Average BMI', labelpad=10)
plt.xlabel('Gender', labelpad=10);

#---------------------------------------------------------------------------
plt.subplot(3,2,2)
sb.pointplot(data=df[df.education == 'college or equivalent'], x='gender', y='bmi', hue='gen_health', linestyles="", ci='sd', 
             dodge=.3, order=gender_order, hue_order=health_order, palette='inferno', errwidth=1.5);

plt.xticks(rotation=0)
plt.title("Education = college or equivalent \nAverage BMI vs General Health for both Genders", fontsize= 15)
plt.ylabel('Average BMI', labelpad=10)
plt.xlabel('Gender', labelpad=10);

#--------------------------------------------------------------------------------
plt.subplot(3,2,3)
sb.pointplot(data=df[df.education == 'secondary or equivalent'], x='gender', y='bmi', hue='gen_health', linestyles="", ci='sd', 
             dodge=.3, order=gender_order, hue_order=health_order, palette='inferno', errwidth=1.5);

plt.xticks(rotation=0)
plt.title("Education = secondary or equivalent \nAverage BMI vs General Health for both Genders", fontsize= 15)
plt.ylabel('Average BMI', labelpad=10)
plt.xlabel('Gender', labelpad=10);

#-----------------------------------------------------------------------------------
plt.subplot(3,2,4)
sb.pointplot(data=df[df.education == 'preparatory'], x='gender', y='bmi', hue='gen_health', linestyles="", ci='sd', 
             dodge=.3, order=gender_order, hue_order=health_order, palette='inferno', errwidth=1.5);

plt.xticks(rotation=0)
plt.title("Education = preparatory \nAverage BMI vs General Health for both Genders", fontsize= 15)
plt.ylabel('Average BMI', labelpad=10)
plt.xlabel('Gender', labelpad=10);
#---------------------------------------------------------------------------------
plt.subplot(3,2,5)
sb.pointplot(data=df[df.education == 'less than preparatory'], x='gender', y='bmi', hue='gen_health', linestyles="", ci='sd', 
             dodge=.3, order=gender_order, hue_order=health_order, palette='inferno', errwidth=1.5);

plt.xticks(rotation=0)
plt.title("Education = less than preparatory \nAverage BMI vs General Health for both Genders", fontsize= 15)
plt.ylabel('Average BMI', labelpad=10)
plt.xlabel('Gender', labelpad=10);
plt.savefig(gender_dir+"/" + "average BMI vs general health by gender vs education level ");
plt.savefig(education_dir+"/" + "average BMI vs general health by gender vs education level");
plt.savefig(genera_health_dir+"/" + "average BMI vs general health by gender vs education level");
plt.savefig(bmi_dir+"/" + "average BMI vs general health by gender vs education level");
#$$$$$$$$$$$$$$$$$$$$$
#	{gender, gen_health, age, weight}
g = sb.FacetGrid(data=df, col='gen_health', col_order=health_order, 
                 row='gender', row_order=gender_order, margin_titles=True, height=5, aspect=1.5)
g.map(sb.regplot, 'age', 'weight_kg', scatter_kws={'alpha':.25}, truncate=False, color='teal');
plt.savefig(gender_dir+"/" + "weight vs age vs general health condition by gender");
plt.savefig(age_dir+"/" + "weight vs age vs general health condition by gender");
plt.savefig(genera_health_dir+"/" + "weight vs age vs general health condition by gender");
plt.savefig(weight_dir+"/" + "weight vs age vs general health condition by gender");
#$$$$$$$$$$$$$$$$$$$$$$
#	{gen_health, bmi, waist, age}
plt.figure(figsize=[20, 16])

plt.subplot(2,3,1)
plt.scatter(data=df[df.gen_health == health_order[0]], x='age', y='bmi', c='waist_cm', cmap='coolwarm', alpha=1)
plt.colorbar(label='');

plt.title('Health = Excellen \nBMI vs Age vs Waist circumference', fontsize= 15)
plt.ylabel('Body Mass Index', labelpad=10)
plt.xlabel('', labelpad=10);
#---------------------------------------------------------------------------
plt.subplot(2,3,2)
plt.scatter(data=df[df.gen_health == health_order[1]], x='age', y='bmi', c='waist_cm', cmap='coolwarm', alpha=1)
plt.colorbar(label='');

plt.title('Health = Very Good \nBMI vs Age vs Waist circumference', fontsize= 15)
plt.ylabel('', labelpad=10)
plt.xlabel('', labelpad=10);

#---------------------------------------------------------------------------
plt.subplot(2,3,3)
plt.scatter(data=df[df.gen_health == health_order[2]], x='age', y='bmi', c='waist_cm', cmap='coolwarm', alpha=1)
plt.colorbar(label='Waist (cm)');

plt.title('Health = Good \nBMI vs Age vs Waist circumference', fontsize= 15)
plt.ylabel('', labelpad=10)
plt.xlabel('Age', labelpad=10);

#---------------------------------------------------------------------------
plt.subplot(2,3,4)
plt.scatter(data=df[df.gen_health == health_order[3]], x='age', y='bmi', c='waist_cm', cmap='coolwarm', alpha=1)
plt.colorbar(label='');

plt.title('Health = Fair \nBMI vs Age vs Waist circumference', fontsize= 15)
plt.ylabel('Body Mass Index', labelpad=10)
plt.xlabel('Age', labelpad=10);

#---------------------------------------------------------------------------
plt.subplot(2,3,5)
plt.scatter(data=df[df.gen_health == health_order[4]], x='age', y='bmi', c='waist_cm', cmap='coolwarm', alpha=1)
plt.colorbar(label='Waist (cm)');

plt.title('Health = Poor \nBMI vs Age vs Waist circumference', fontsize= 15)
plt.ylabel('', labelpad=10)
plt.xlabel('Age', labelpad=10);
plt.savefig(bmi_dir+"/" + "bmi vs waist vs age vs general health condition");
plt.savefig(age_dir+"/" + "bmi vs waist vs age vs general health condition");
plt.savefig(genera_health_dir+"/" + "bmi vs waist vs age vs general health condition");
plt.savefig(waist_dir+"/" + "bmi vs waist vs age vs general health condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$
#		{{{{{Diseases Analytics}}}}}
#	{asthma}
plt.figure(figsize=[15,10])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='asthma', palette = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']);
_counts = df.asthma.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = _counts[label.get_text()]
    count_string = '{}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(_counts[0]/100), count_string, ha = 'center', color = 'black')

plt.title('Asthma Disease Distribution', fontsize= 20, pad=20)
plt.xlabel('Asthma patient?', fontsize= 15, labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']
_counts = _counts.sort_index()

if (round(_counts.values.min()*100 / df.shape[0], 1) < 5):
    labels_dis = 1.25
    prc_dis = 1.1
else:
    labels_dis = 1.1
    prc_dis = .7

plt.pie(x=_counts, labels=_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },  pctdistance=prc_dis, labeldistance=labels_dis)#, textprops={'color':'white'}

plt.axis('square')
plt.xlabel('Asthma patient?', fontsize= 15, labelpad=40)
plt.title('Asthma Disease Proportions', fontsize= 20, pad=60);
plt.savefig(asthma_dir+"/" + "Asthma Disease Proportions");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{chf Congestive heart failure}
plt.figure(figsize=[15,10])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='chf', palette = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']);
_counts = df.chf.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = _counts[label.get_text()]
    count_string = '{}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(_counts[0]/100), count_string, ha = 'center', color = 'black') #, palette=()

plt.title('Condition of Heart Failure Distribution', fontsize= 17, pad=20)
plt.xlabel('CHF patient?', fontsize= 13, labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']
_counts = _counts.sort_index()

if (round(_counts.values.min()*100 / df.shape[0], 1) < 5):
    labels_dis = 1.25
    prc_dis = 1.1
else:
    labels_dis = 1.1
    prc_dis = .7

plt.pie(x=_counts, labels=_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },  pctdistance=prc_dis, labeldistance=labels_dis)#, textprops={'color':'white'}

plt.axis('square')
plt.xlabel('CHF patient?', fontsize= 13, labelpad=40)
plt.title('Cognitive Heart Failure Proportions', fontsize= 17, pad=60);
plt.savefig(chf_dir+"/" + "Cognitive Heart Failure Proportions");
#$$$$$$$$$$$$$$$$$$$$$$$$
#	{cad Coronary artery disease}
plt.figure(figsize=[12,8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='cad', palette = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']);
_counts = df.cad.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = _counts[label.get_text()]
    count_string = '{}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(_counts[0]/100), count_string, ha = 'center', color = 'black') #, palette=()

plt.title('Coronary Artery Disease Distribution', fontsize= 17, pad=20)
plt.xlabel('CAD patient?', fontsize= 13, labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']
_counts = _counts.sort_index()

if (round(_counts.values.min()*100 / df.shape[0], 1) < 5):
    labels_dis = 1.25
    prc_dis = 1.1
else:
    labels_dis = 1.1
    prc_dis = .7

plt.pie(x=_counts, labels=_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },  pctdistance=prc_dis, labeldistance=labels_dis)#, textprops={'color':'white'}

plt.axis('square')

plt.title('Coronary Artery Disease Proportions', fontsize= 17, pad=60)
plt.xlabel('CAD patient?', fontsize= 13, labelpad=40);
plt.savefig(cad_dir+"/" + "Coronary Artery Disease Proportions");
#$$$$$$$$$$$$$$$$$$$$$$
#	{mi Myocardial infarction}
plt.figure(figsize=[15,8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='mi', palette = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']);
_counts = df.mi.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = _counts[label.get_text()]
    count_string = '{}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(_counts[0]/100), count_string, ha = 'center', color = 'black') #, palette=()

plt.title('Myocardial Infarction Disease Distribution', fontsize= 17, pad=20)
plt.xlabel('MI patient?', fontsize= 13, labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']
_counts = _counts.sort_index()

if (round(_counts.values.min()*100 / df.shape[0], 1) < 5):
    labels_dis = 1.25
    prc_dis = 1.1
else:
    labels_dis = 1.1
    prc_dis = .7
    
plt.pie(x=_counts, labels=_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },  pctdistance=prc_dis, labeldistance=labels_dis)#, textprops={'color':'white'}

plt.axis('square')

plt.title('Myocardial Infarction Disease Proportions', fontsize= 17, pad=60)
plt.xlabel('MI patient?', fontsize= 13, labelpad=40);
plt.savefig(mi_dir+"/" + "Myocardial Infarction Disease Proportions");
#$$$$$$$$$$$$$$$$$
#	{cva Cerebrovascular Accident (Stroke)}
plt.figure(figsize=[12,8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='cva', palette = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']);
_counts = df.cva.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = _counts[label.get_text()]
    count_string = '{}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(_counts[0]/100), count_string, ha = 'center', color = 'black') #, palette=()

plt.title('Cerebrovascular Accident patients Distribution', fontsize= 17, pad=20)
plt.xlabel('CVA patient?', fontsize= 13, labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']
_counts = _counts.sort_index()

if (round(_counts.values.min()*100 / df.shape[0], 1) < 5):
    labels_dis = 1.25
    prc_dis = 1.1
else:
    labels_dis = 1.1
    prc_dis = .7
    
plt.pie(x=_counts, labels=_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },  pctdistance=prc_dis, labeldistance=labels_dis)#, textprops={'color':'white'}

plt.axis('square')

plt.title('Cerebrovascular Accident patients Proportions', fontsize= 17, pad=60)
plt.xlabel('CVA patient?', fontsize= 13, labelpad=40);
plt.savefig(cva_dir+"/" + "Cerebrovascular Accident patients Proportions");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{copd Chronic obstructive pulmonary disease}
plt.figure(figsize=[12,8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='copd', palette = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']);
_counts = df.copd.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = _counts[label.get_text()]
    count_string = '{}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(_counts[0]/100), count_string, ha = 'center', color = 'black') #, palette=()

plt.title('Chronic Obstructive Pulmonary patients Distribution', fontsize= 17, pad=20)
plt.xlabel('COPD patient?', fontsize= 13, labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']
_counts = _counts.sort_index()

if (round(_counts.values.min()*100 / df.shape[0], 1) < 5):
    labels_dis = 1.25
    prc_dis = 1.1
else:
    labels_dis = 1.1
    prc_dis = .7
    
plt.pie(x=_counts, labels=_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },  pctdistance=prc_dis, labeldistance=labels_dis)#, textprops={'color':'white'}

plt.axis('square')

plt.title('Chronic Obstructive Pulmonary patients Proportions', fontsize= 17, pad=60)
plt.xlabel('COPD patient?', fontsize= 13, labelpad=40);
plt.savefig(copd_dir+"/" + "Chronic Obstructive Pulmonary patients Proportions");
#$$$$$$$$$$$$$$$$$$$$$$$$$
#	{cancer}
plt.figure(figsize=[12,8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='cancer', palette = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']);
_counts = df.cancer.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = _counts[label.get_text()]
    count_string = '{}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(_counts[0]/100), count_string, ha = 'center', color = 'black') #, palette=()

plt.title('Cancer patients Distribution', fontsize= 17, pad=20)
plt.xlabel('Cancer patient?', fontsize= 13, labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = [(1,0,0,.7), (.3,1,0,.99), 'powderblue']
_counts = _counts.sort_index()

if (round(_counts.values.min()*100 / df.shape[0], 1) < 5):
    labels_dis = 1.25
    prc_dis = 1.1
else:
    labels_dis = 1.1
    prc_dis = .7
    
plt.pie(x=_counts, labels=_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },  pctdistance=prc_dis, labeldistance=labels_dis)#, textprops={'color':'white'}

plt.axis('square')

plt.title('Cancer patients Proportions', fontsize= 17, pad=60)
plt.xlabel('Cancer patient?', fontsize= 13, labelpad=40);
plt.savefig(cancer_dir+"/" + "Cancer patients Proportions");
#$$$$$$$$$$$$$$$$$$$$$$$$$
#	{hypertension}
plt.figure(figsize=[12,8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='hypertension', palette = [(1,0,0,.7), (.3,1,0,.99), 'yellow', 'powderblue']);
_counts = df.hypertension.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = _counts[label.get_text()]
    count_string = '{}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(_counts[0]/100), count_string, ha = 'center', color = 'black') #, palette=()

plt.title('Hypertension patients Distribution', fontsize= 17, pad=20)
plt.xlabel('Hypertension patient?', fontsize= 13, labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = [(1,0,0,.7), (.3,1,0,.99), 'yellow', 'powderblue']
_counts = _counts.sort_index()

if (round(_counts.values.min()*100 / df.shape[0], 1) < 5):
    labels_dis = 1.25
    prc_dis = 1.1
else:
    labels_dis = 1.1
    prc_dis = .7
    
plt.pie(x=_counts, labels=_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' }, pctdistance=prc_dis, labeldistance=labels_dis)#, textprops={'color':'white'}

plt.axis('square')

plt.title('Hypertension patients Proportions', fontsize= 17, pad=60)
plt.xlabel('Hypertension patient?', fontsize= 13, labelpad=40);
plt.savefig(hypertension_dir+"/" + "Hypertension patients Proportions");
#$$$$$$$$$$$$$$$$$$$$$$
#	{diabetes}
plt.figure(figsize=[12,8])
sb.set_theme(style="darkgrid")

#_______________________________first plot____________________________________
plt.subplot(1,2,1)

sb.countplot(data=df, x='diabetes', palette = [(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'powderblue']);#(1,0,0,.7), (.3,1,0,.99), 'powderblue'
_counts = df.diabetes.value_counts()
locs, labels = plt.xticks()

# loop through each pair of locations and labels
for loc, label in zip(locs, labels):

    # get the text property for the label to get the correct count
    count = _counts[label.get_text()]
    count_string = '{}'.format(count)

    # print the annotation just above the top of the bar
    plt.text(loc, count+(_counts[0]/100), count_string, ha = 'center', color = 'black') #, palette=()

plt.title('Diabetic patients Distribution', fontsize= 17, pad=20)
plt.xlabel('Diabetic patient?', fontsize= 13, labelpad=20)

#__________________________________second plot________________________________________
plt.subplot(1,2,2)

colors = [(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'powderblue']
_counts = _counts.sort_index()

if (round(_counts.values.min()*100 / df.shape[0], 1) < 5):
    labels_dis = 1.25
    prc_dis = 1.1
else:
    labels_dis = 1.1
    prc_dis = .7
    
plt.pie(x=_counts, labels=_counts.index, startangle=90, counterclock=True, colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p), 
        wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' },  pctdistance=prc_dis, labeldistance=labels_dis)#, textprops={'color':'white'}

plt.axis('square')

plt.title('Diabetic patients Proportions', fontsize= 17, pad=60)
plt.xlabel('Diabetic patient?', fontsize= 13, labelpad=40);
plt.savefig(diabetes_dir+"/" + "Diabetic patients Proportions");
#$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[16,8])
sb.set_theme(style='whitegrid')

#-------------------------------board--------------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
sb.kdeplot(data=df, x='pulse', cut=0, fill= True, color="#00AFBB");
plt.title('Mean Pulse Distribution Density', fontsize= 15)
plt.ylabel('Density', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.25)



plt.text(115, .025, 'dangerous\nHigh pulse', ha = 'center', color = 'red', fontsize=15, alpha=.7)
plt.text(85, .015, 'Normal\npulse', ha = 'center', color = 'b', fontsize=15, alpha=1)
plt.text(50, .005, 'dangerous\nLow pulse', ha = 'center', color = 'red', fontsize=15, alpha=.7)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);
plt.savefig(pulse_dir+"/" + "Mean Pulse Distribution Density");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[16,8])
sb.set_theme(style='whitegrid')

# ------------------------------------------board----------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50, 45]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df, x='pulse', bins= bins, color="blue", alpha=.5);
plt.title('Patients Pulse Distribution', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.25)



plt.text(115, df.shape[0]/400, 'dangerous\nHigh pulse', ha = 'center', color = 'red', fontsize=15, alpha=.5)
plt.text(85, df.shape[0]/600, 'Normal\npulse', ha = 'center', color = 'b', fontsize=15, alpha=.99)
plt.text(50, df.shape[0]/1000, 'dangerous\nLow pulse', ha = 'center', color = 'red', fontsize=15, alpha=.5)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);
plt.savefig(pulse_dir+"/" + "Patients Pulse Distribution");
#$$$$$$$$$$$$$$$$$$$$$$
#	{systolic blood pressure}
plt.figure(figsize=[20,13])
sb.set_theme(style='whitegrid')


#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.2)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.2)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.1)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------

bins = np.arange(65, 252, 2)

plt.hist(data=df, x='sys_bp', bins= bins, color="b", alpha=1);
plt.title('Patients Systolic Blood Pressure Distribution', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Systolic BP', labelpad=10);

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(65, 250);
plt.savefig(sys_bp_dir+"/" + "Patients Systolic Blood Pressure Distribution");
#$$$$$$$$$$$

plt.figure(figsize=[6,4])
sb.set_theme(style='white')
my_palette = [(1,0,0,.3), (0.3,1,.4,.5), (.99,.99,.10,.45), (1,.8,0,.7), (1,0,0,.1), (1,0,0,.35)]
sb.palplot(my_palette, size=1.7)
plt.xlabel('\nSystolic Blood Pressure Ranges')
plt.xticks([0,1,2,3,4,5], ['Hypotension', 'Normal', 'Elevated', 'Hypertension_1', 'Hypertension_2', '  Hypertension crisis']);
plt.savefig(sys_bp_dir+"/" + "Systolic Blood Pressure Ranges");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,10])
sb.set_theme(style='whitegrid')
#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.2)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.2)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.1)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------

sb.kdeplot(data=df, x='sys_bp', cut=0, fill= False, color="#00AFBB");
plt.title('Patients Systolic Blood Pressure Distribution', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Systolic BP', labelpad=10);

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(65, 250);
plt.savefig(sys_bp_dir+"/" + "Mean systolic BP Distribution Density");
#$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,10])

#_______________________________first plot____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.2)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.1)

sb.kdeplot(data=df, x='dia_bp', cut=0, fill= True, color="#00AFBB");
plt.title('Mean Diastolic BP Distribution Density', fontsize= 15)
plt.ylabel('Density', labelpad=10)
plt.xlabel('Mean Diastolic BP', labelpad=10);

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.99)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=.99)

ticks = np.arange(30, 160, 10)
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(30, 150);
plt.savefig(dia_bp_dir+"/" + "Mean Diastolic BP Distribution Density");
#$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,10])
sb.set_theme(style='whitegrid')

#_______________________________board____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.2)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.1)
#----------------------------------------------------------------------

bins = np.arange(30, 152, 2)

plt.hist(data=df, x='dia_bp', bins= bins, color="b", alpha=1);
plt.title('Patients Diastolic Blood Pressure Distribution', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Diastolic BP', labelpad=10);

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='w', alpha=.99)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(30, 150);
plt.savefig(dia_bp_dir+"/" + "Patients Diastolic Blood Pressure Distribution");
#$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[6,4])
sb.set_theme(style='white')
my_palette = [(1,0,0,.3), (0.3,1,.4,.2), (1,.99,0,.3), (1,0,0,.1), (1,0,0,.35)]
sb.palplot(my_palette, size=1.7)
plt.xlabel('\nDiastolic Blood Pressure Ranges')
plt.xticks([0,1,2,3,4], ['Hypotension', 'Normal', 'Hypertension_1', 'Hypertension_2', '  Hypertension crisis']);
plt.savefig(dia_bp_dir+"/" + "Diastolic Blood Pressure Ranges");
#$$$$$$$$$$$$$$$$$$$$$$$$$
#		{Bivariate visuals (Diseases)}
#	{asthma}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='asthma', hue='chf', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('Asthma vs CHF', fontsize= 20, pad=10)
plt.xlabel('Had Asthma ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(asthma_dir+"/" + "Asthma vs CHF");
plt.savefig(chf_dir+"/" + "Asthma vs CHF");
#$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='asthma', hue='cad', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('Asthma vs CAD', fontsize= 20, pad=10)
plt.xlabel('Had Asthma ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(asthma_dir+"/" + "Asthma vs CAD");
plt.savefig(cad_dir+"/" + "Asthma vs CAD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='asthma', hue='mi', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('Asthma vs MI', fontsize= 20, pad=10)
plt.xlabel('Had Asthma ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(asthma_dir+"/" + "Asthma vs MI");
plt.savefig(mi_dir+"/" + "Asthma vs MI");
#$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='asthma', hue='cva', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('Asthma vs CVA', fontsize= 20, pad=10)
plt.xlabel('Had Asthma ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(asthma_dir+"/" + "Asthma vs CVA");
plt.savefig(cva_dir+"/" + "Asthma vs CVA");
#$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='asthma', hue='copd', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('Asthma vs COPD', fontsize= 20, pad=10)
plt.xlabel('Had Asthma ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(asthma_dir+"/" + "Asthma vs COPD");
plt.savefig(copd_dir+"/" + "Asthma vs COPD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='asthma', hue='cancer', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('Asthma vs Cancer', fontsize= 20, pad=10)
plt.xlabel('Had Asthma ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(asthma_dir+"/" + "Asthma vs Cancer");
plt.savefig(cancer_dir+"/" + "Asthma vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='asthma', hue='hypertension', palette=[(1,0,0,.7), (.3,1,0,.99), 'yellow', 'blue']);

plt.title('Asthma vs Hypertension', fontsize= 20, pad=10)
plt.xlabel('Had Asthma ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(asthma_dir+"/" + "Asthma vs Hypertension");
plt.savefig(hypertension_dir+"/" + "Asthma vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='asthma', hue='diabetes', palette=[(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'blue']);

plt.title('Asthma vs Diabetes', fontsize= 20, pad=10)
plt.xlabel('Had Asthma ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(asthma_dir+"/" + "Asthma vs Diabetes");
plt.savefig(diabetes_dir+"/" + "Asthma vs Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

plt.figure(figsize=[20,15])
sb.set_theme(style='whitegrid')


plt.subplot(2,1,1)
#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df[df.asthma == 'no'], x='pulse', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.asthma == 'yes'], x='pulse', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.asthma == 'unknown'], x='pulse', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='Had Athma?', shadow=True)

plt.title('Patients Pulse Distribution vs Asthma', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);


plt.subplot(2,1,2)
# ------------------------------------------board----------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50, 45]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)
#----------------------------------------------------------
df['pulse'][df.asthma == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='limegreen');
df['pulse'][df.asthma == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '3', fontsize=20,
                                                                color='red');
df['pulse'][df.asthma == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='Had Asthma?', loc='upper right')

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 130, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Pulse Distribution vs Asthma', fontsize= 25)
plt.xlabel('Mean Pulse', fontsize= 20, labelpad=10)
plt.ylabel('Count', fontsize= 15)

plt.xlim(40,130);

plt.savefig(asthma_dir+"/" + "Patients Pulse Distribution vs Asthma");
plt.savefig(pulse_dir+"/" + "Patients Pulse Distribution vs Asthma");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')


plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(65, 250+2, 2)

plt.hist(data=df[df.asthma == 'no'], x='sys_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.asthma == 'yes'], x='sys_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.asthma == 'unknown'], x='sys_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='Had Athma?', shadow=True)

plt.title('Patients Systolic BP Distribution vs Asthma', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Systolic BP', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs Asthma', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(65, 250);



plt.subplot(2,1,2)
#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.13)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.13)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.07)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------
df['sys_bp'][df.asthma == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['sys_bp'][df.asthma == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');

df['sys_bp'][df.asthma == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='Had Asthma?', loc='upper right')

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs Asthma', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(65, 250);
plt.savefig(asthma_dir+"/" + "Systolic Blood Pressure Distribution vs Asthma");
plt.savefig(sys_bp_dir+"/" + "Systolic Blood Pressure Distribution vs Asthma");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------
#___________________________________________________________________
bins = np.arange(30, 150+2, 2)

plt.hist(data=df[df.asthma == 'no'], x='dia_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.asthma == 'yes'], x='dia_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.asthma == 'unknown'], x='dia_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='Had Athma?', shadow=True)

plt.title('Patients Diastolic BP Distribution vs Asthma', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Diastolic BP', labelpad=10);

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs Asthma', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(30, 150);

#_______________________________second plot____________________________________
plt.subplot(2,1,2)
#_______________________________board____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.15)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.15)
#----------------------------------------------------------------------

df['dia_bp'][df.asthma == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['dia_bp'][df.asthma == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');
df['dia_bp'][df.asthma == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='Had Asthma?', loc='upper right')

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs Asthma', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(30, 150);
plt.savefig(asthma_dir+"/" + "Diastolic Blood Pressure Distribution vs Asthma");
plt.savefig(dia_bp_dir+"/" + "Diastolic Blood Pressure Distribution vs Asthma");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{chf Cognitive heart failure}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='chf', hue='cad', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CHF vs CAD', fontsize= 20, pad=10)
plt.xlabel('CHF ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(chf_dir+"/" + "CHF vs CAD");
plt.savefig(cad_dir+"/" + "CHF vs CAD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='chf', hue='mi', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CHF vs MI', fontsize= 20, pad=10)
plt.xlabel('CHF ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(chf_dir+"/" + "CHF vs MI");
plt.savefig(mi_dir+"/" + "CHF vs MI");
#$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='chf', hue='cva', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CHF vs CVA', fontsize= 20, pad=10)
plt.xlabel('CHF ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(chf_dir+"/" + "CHF vs CVA");
plt.savefig(cva_dir+"/" + "CHF vs CVA");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='chf', hue='copd', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CHF vs COPD', fontsize= 20, pad=10)
plt.xlabel('CHF ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(chf_dir+"/" + "CHF vs COPD");
plt.savefig(copd_dir+"/" + "CHF vs COPD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='chf', hue='cancer', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CHF vs Cancer', fontsize= 20, pad=10)
plt.xlabel('CHF ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(chf_dir+"/" + "CHF vs Cancer");
plt.savefig(cancer_dir+"/" + "CHF vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='chf', hue='hypertension', palette=[(1,0,0,.7), (.3,1,0,.99), 'yellow', 'blue']);

plt.title('CHF vs Hypertension', fontsize= 20, pad=10)
plt.xlabel('CHF ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(chf_dir+"/" + "CHF vs Hypertension");
plt.savefig(hypertension_dir+"/" + "CHF vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='chf', hue='diabetes', palette=[(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'blue']);

plt.title('CHF vs Diabetes', fontsize= 20, pad=10)
plt.xlabel('CHF ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(chf_dir+"/" + "CHF vs Diabetes");
plt.savefig(diabetes_dir+"/" + "CHF vs Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14,15])
sb.set_theme(style='whitegrid')


plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df[df.chf == 'no'], x='pulse', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.chf == 'yes'], x='pulse', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.chf == 'unknown'], x='pulse', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='CHF?', shadow=True)

plt.title('Patients Pulse Distribution vs CHF', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 140, 10)
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);


plt.subplot(2,1,2)
# ------------------------------------------board----------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50, 45]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)

#----------------------------------------------------------
df['pulse'][df.chf == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='limegreen');
df['pulse'][df.chf == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '3', fontsize=20,
                                                                color='red');

df['pulse'][df.chf == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='CHF', loc='upper right', title_fontsize=15)

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 130, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Pulse Distribution vs CHF', fontsize= 25)
plt.xlabel('Mean Pulse', fontsize= 20, labelpad=10)
plt.ylabel('Count', fontsize= 15)

plt.xlim(40,130);
plt.savefig(chf_dir+"/" + "Pulse Distribution vs CHF");
plt.savefig(pulse_dir+"/" + "Pulse Distribution vs CHF");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(65, 250+2, 2)

plt.hist(data=df[df.chf == 'no'], x='sys_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.chf == 'yes'], x='sys_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.chf == 'unknown'], x='sys_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='CHF?', shadow=True)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs CHF', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(65, 250);


#==================================second plot===============================
plt.subplot(2,1,2)
#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.13)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.13)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.07)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------
df['sys_bp'][df.chf == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['sys_bp'][df.chf == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');

df['sys_bp'][df.chf == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='CHF', loc='upper right', title_fontsize=15)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs CHF', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(65, 250);
plt.savefig(chf_dir+"/" + "Systolic Blood Pressure Distribution vs CHF");
plt.savefig(sys_bp_dir+"/" + "Systolic Blood Pressure Distribution vs CHF");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------
#___________________________________________________________________
bins = np.arange(30, 150+2, 2)

plt.hist(data=df[df.chf == 'no'], x='dia_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.chf == 'yes'], x='dia_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.chf == 'unknown'], x='dia_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='CHF?', shadow=True)

plt.title('Patients Diastolic BP Distribution vs CHF', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Diastolic BP', labelpad=10);

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs CHF', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(30, 150);

#_______________________________second plot____________________________________
plt.subplot(2,1,2)
#_______________________________board____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.15)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.15)
#----------------------------------------------------------------------

df['dia_bp'][df.chf == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['dia_bp'][df.chf == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');
df['dia_bp'][df.chf == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='CHF?', loc='upper right')

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs CHF', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(30, 150);
plt.savefig(chf_dir+"/" + "Diastolic Blood Pressure Distribution vs CHF");
plt.savefig(dia_bp_dir+"/" + "Diastolic Blood Pressure Distribution vs CHF");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{cad coronary artery disease}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='cad', hue='mi', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CAD Distribution vs MI', fontsize= 20, pad=10)
plt.xlabel('CAD ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cad_dir+"/" + "CAD Distribution vs MI");
plt.savefig(mi_dir+"/" + "CAD Distribution vs MI");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='cad', hue='cva', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CAD Distribution vs CVA', fontsize= 20, pad=10)
plt.xlabel('CAD ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cad_dir+"/" + "CAD Distribution vs CVA");
plt.savefig(cva_dir+"/" + "CAD Distribution vs CVA");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='cad', hue='copd', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CAD Distribution vs COPD', fontsize= 20, pad=10)
plt.xlabel('CAD ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cad_dir+"/" + "CAD Distribution vs COPD");
plt.savefig(copd_dir+"/" + "CAD Distribution vs COPD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='cad', hue='cancer', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CAD Distribution vs Cancer', fontsize= 20, pad=10)
plt.xlabel('CAD ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cad_dir+"/" + "CAD Distribution vs Cancer");
plt.savefig(cancer_dir+"/" + "CAD Distribution vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='cad', hue='hypertension', palette=[(1,0,0,.7), (.3,1,0,.99), 'yellow', 'blue']);

plt.title('CAD Distribution vs Hypertension', fontsize= 20, pad=10)
plt.xlabel('CAD ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cad_dir+"/" + "CAD Distribution vs Hypertension");
plt.savefig(hypertension_dir+"/" + "CAD Distribution vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='cad', hue='diabetes', palette=[(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'blue']);

plt.title('CAD Distribution vs Diabetes', fontsize= 20, pad=10)
plt.xlabel('CAD ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cad_dir+"/" + "CAD Distribution vs Diabetes");
plt.savefig(diabetes_dir+"/" + "CAD Distribution vs Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,15])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df[df.cad == 'no'], x='pulse', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.cad == 'yes'], x='pulse', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.cad == 'unknown'], x='pulse', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='CAD?', shadow=True)

plt.title('Patients Pulse Distribution vs CAD', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);


#_______________________________second plot____________________________________
plt.subplot(2,1,2)
# ------------------------------------------board----------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50, 45]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)

#----------------------------------------------------------
df['pulse'][df.cad == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='limegreen');
df['pulse'][df.cad == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '3', fontsize=20,
                                                                color='red');

df['pulse'][df.cad == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='CAD', loc='upper right', title_fontsize=15)

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 130, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Pulse Distribution vs CAD', fontsize= 25)
plt.xlabel('Mean Pulse', fontsize= 20, labelpad=10)
plt.ylabel('Count', fontsize= 15)

plt.xlim(40,130);
plt.savefig(cad_dir+"/" + "Pulse Distribution vs CAD");
plt.savefig(pulse_dir+"/" + "Pulse Distribution vs CAD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(65, 250+2, 2)

plt.hist(data=df[df.cad == 'no'], x='sys_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.cad == 'yes'], x='sys_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.cad == 'unknown'], x='sys_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='CAD?', shadow=True)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs CAD', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(65, 250);


#==================================second plot===============================
plt.subplot(2,1,2)
#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.13)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.13)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.07)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------
df['sys_bp'][df.cad == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['sys_bp'][df.cad == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');

df['sys_bp'][df.cad == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='CAD', loc='upper right', title_fontsize=15)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs CAD', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(65, 250);
plt.savefig(cad_dir+"/" + "Systolic Blood Pressure Distribution vs CAD");
plt.savefig(sys_bp_dir+"/" + "Systolic Blood Pressure Distribution vs CAD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------
#___________________________________________________________________
bins = np.arange(30, 150+2, 2)

plt.hist(data=df[df.cad == 'no'], x='dia_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.cad == 'yes'], x='dia_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.cad == 'unknown'], x='dia_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='CAD?', shadow=True)

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs CAD', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);

#_______________________________second plot____________________________________
plt.subplot(2,1,2)
#_______________________________board____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.15)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.15)
#----------------------------------------------------------------------

df['dia_bp'][df.cad == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['dia_bp'][df.cad == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');
df['dia_bp'][df.cad == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');
plt.legend(['no','yes', 'unknown'], title='CAD?', loc='upper right')

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs CAD', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);
plt.savefig(cad_dir+"/" + "Diastolic Blood Pressure Distribution vs CAD");
plt.savefig(dia_bp_dir+"/" + "Diastolic Blood Pressure Distribution vs CAD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{MI}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='mi', hue='cva', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('MI Distribution vs CVA', fontsize= 20, pad=10)
plt.xlabel('MI ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(mi_dir+"/" + "MI Distribution vs CVA");
plt.savefig(cva_dir+"/" + "MI Distribution vs CVA");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='mi', hue='copd', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('MI Distribution vs COPD', fontsize= 20, pad=10)
plt.xlabel('MI ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(mi_dir+"/" + "MI Distribution vs COPD");
plt.savefig(copd_dir+"/" + "MI Distribution vs COPD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='mi', hue='cancer', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('MI Distribution vs Cancer', fontsize= 20, pad=10)
plt.xlabel('MI ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(mi_dir+"/" + "MI Distribution vs Cancer");
plt.savefig(cancer_dir+"/" + "MI Distribution vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='mi', hue='hypertension', palette=[(1,0,0,.7), (.3,1,0,.99), 'yellow', 'blue']);

plt.title('MI Distribution vs Hypertension', fontsize= 20, pad=10)
plt.xlabel('MI ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(mi_dir+"/" + "MI Distribution vs Hypertension");
plt.savefig(hypertension_dir+"/" + "MI Distribution vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[15,8])
sb.countplot(data=df, x='mi', hue='diabetes', palette=[(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'blue']);

plt.title('MI Distribution vs Diabetes', fontsize= 20, pad=10)
plt.xlabel('MI ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(mi_dir+"/" + "MI Distribution vs Diabetes");
plt.savefig(diabetes_dir+"/" + "MI Distribution vs Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,15])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df[df.mi == 'no'], x='pulse', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.mi == 'yes'], x='pulse', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.mi == 'unknown'], x='pulse', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='MI?', shadow=True)

plt.title('Patients Pulse Distribution vs MI', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);


#_______________________________second plot____________________________________
plt.subplot(2,1,2)
# ------------------------------------------board----------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50, 45]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)

#----------------------------------------------------------
df['pulse'][df.mi == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='limegreen');
df['pulse'][df.mi == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '3', fontsize=20,
                                                                color='red');
df['pulse'][df.mi == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='MI', loc='upper right', title_fontsize=15)

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 130, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Pulse Distribution vs MI', fontsize= 25)
plt.xlabel('Mean Pulse', fontsize= 20, labelpad=10)
plt.ylabel('Count', fontsize= 15)

plt.xlim(40,130);
plt.savefig(mi_dir+"/" + "Pulse Distribution vs MI");
plt.savefig(pulse_dir+"/" + "Pulse Distribution vs MI");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(65, 250+2, 2)

plt.hist(data=df[df.mi == 'no'], x='sys_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.mi == 'yes'], x='sys_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.mi == 'unknown'], x='sys_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='MI?', shadow=True)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs MI', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(65, 250);


#==================================second plot===============================
plt.subplot(2,1,2)
#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.13)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.13)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.07)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------
df['sys_bp'][df.mi == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['sys_bp'][df.mi == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');

df['sys_bp'][df.mi == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='MI', loc='upper right', title_fontsize=15)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs MI', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(65, 250);
plt.savefig(mi_dir+"/" + "Systolic Blood Pressure Distribution vs MI");
plt.savefig(sys_bp_dir+"/" + "Systolic Blood Pressure Distribution vs MI");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------
#___________________________________________________________________
bins = np.arange(30, 150+2, 2)

plt.hist(data=df[df.mi == 'no'], x='dia_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.mi == 'yes'], x='dia_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.mi == 'unknown'], x='dia_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='MI?', shadow=True)

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs MI', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);

#_______________________________second plot____________________________________
plt.subplot(2,1,2)
#_______________________________board____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.15)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.15)
#----------------------------------------------------------------------

df['dia_bp'][df.mi == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['dia_bp'][df.mi == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');
df['dia_bp'][df.mi == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');
plt.legend(['no','yes', 'unknown'], title='MI?', loc='upper right')

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs MI', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);
plt.savefig(mi_dir+"/" + "Diastolic Blood Pressure Distribution vs MI");
plt.savefig(dia_bp_dir+"/" + "Diastolic Blood Pressure Distribution vs MI");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{CVA}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='cva', hue='copd', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CVA Distribution vs COPD', fontsize= 20, pad=10)
plt.xlabel('CVA ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cva_dir+"/" + "CVA Distribution vs COPD");
plt.savefig(copd_dir+"/" + "CVA Distribution vs COPD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='cva', hue='cancer', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('CVA Distribution vs Cancer', fontsize= 20, pad=10)
plt.xlabel('CVA ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cva_dir+"/" + "CVA Distribution vs Cancer");
plt.savefig(cancer_dir+"/" + "CVA Distribution vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='cva', hue='hypertension', palette=[(1,0,0,.7), (.3,1,0,.99), 'yellow', 'blue']);

plt.title('CVA Distribution vs Hypertension', fontsize= 20, pad=10)
plt.xlabel('CVA ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cva_dir+"/" + "CVA Distribution vs Hypertension");
plt.savefig(hypertension_dir+"/" + "CVA Distribution vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='cva', hue='diabetes', palette=[(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'blue']);

plt.title('CVA Distribution vs Diabetes', fontsize= 20, pad=10)
plt.xlabel('CVA ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cva_dir+"/" + "CVA Distribution vs Diabetes");
plt.savefig(diabetes_dir+"/" + "CVA Distribution vs Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,15])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df[df.cva == 'no'], x='pulse', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.cva == 'yes'], x='pulse', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.cva == 'unknown'], x='pulse', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='CVA?', shadow=True)

plt.title('Patients Pulse Distribution vs CVA', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);


#_______________________________second plot____________________________________
plt.subplot(2,1,2)
# ------------------------------------------board----------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50, 45]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)

#----------------------------------------------------------
df['pulse'][df.cva == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='limegreen');
df['pulse'][df.cva == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '3', fontsize=20,
                                                                color='red');

df['pulse'][df.cva == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='CVA', loc='upper right', title_fontsize=15)

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 130, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Pulse Distribution vs CVA', fontsize= 25)
plt.xlabel('Mean Pulse', fontsize= 20, labelpad=10)
plt.ylabel('Count', fontsize= 15)

plt.xlim(40,130);
plt.savefig(cva_dir+"/" + "Pulse Distribution vs CVA");
plt.savefig(pulse_dir+"/" + "Pulse Distribution vs CVA");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(65, 250+2, 2)

plt.hist(data=df[df.cva == 'no'], x='sys_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.cva == 'yes'], x='sys_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.cva == 'unknown'], x='sys_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='CVA?', shadow=True)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs CVA', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(65, 250);


#==================================second plot===============================
plt.subplot(2,1,2)
#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.13)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.13)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.07)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------
df['sys_bp'][df.cva == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['sys_bp'][df.cva == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');

df['sys_bp'][df.cva == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='CVA', loc='upper right', title_fontsize=15)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs CVA', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(65, 250);
plt.savefig(cva_dir+"/" + "Systolic Blood Pressure Distribution vs CVA");
plt.savefig(sys_bp_dir+"/" + "Systolic Blood Pressure Distribution vs CVA");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------
#___________________________________________________________________
bins = np.arange(30, 150+2, 2)

plt.hist(data=df[df.cva == 'no'], x='dia_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.cva == 'yes'], x='dia_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.cva == 'unknown'], x='dia_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='CVA?', shadow=True)

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs CVA', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);

#_______________________________second plot____________________________________
plt.subplot(2,1,2)
#_______________________________board____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.15)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.15)
#----------------------------------------------------------------------

df['dia_bp'][df.cva == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['dia_bp'][df.cva == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');
df['dia_bp'][df.cva == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');
plt.legend(['no','yes', 'unknown'], title='CVA?', loc='upper right')

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs CVA', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);
plt.savefig(cva_dir+"/" + "Diastolic Blood Pressure Distribution vs CVA");
plt.savefig(dia_bp_dir+"/" + "Diastolic Blood Pressure Distribution vs CVA");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{COPD}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='copd', hue='cancer', palette=[(1,0,0,.7), (.3,1,0,.99), 'blue']);

plt.title('COPD Distribution vs Cancer', fontsize= 20, pad=10)
plt.xlabel('COPD ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(copd_dir+"/" + "COPD Distribution vs Cancer");
plt.savefig(cancer_dir+"/" + "COPD Distribution vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='copd', hue='hypertension', palette=[(1,0,0,.7), (.3,1,0,.99), 'yellow', 'blue']);

plt.title('COPD Distribution vs Hypertension', fontsize= 20, pad=10)
plt.xlabel('COPD ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(copd_dir+"/" + "COPD Distribution vs Hypertension");
plt.savefig(hypertension_dir+"/" + "COPD Distribution vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='copd', hue='diabetes', palette=[(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'blue']);

plt.title('COPD Distribution vs Diabetes', fontsize= 20, pad=10)
plt.xlabel('COPD ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(copd_dir+"/" + "COPD Distribution vs Diabetes");
plt.savefig(diabetes_dir+"/" + "COPD Distribution vs Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,15])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df[df.copd == 'no'], x='pulse', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.copd == 'yes'], x='pulse', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.copd == 'unknown'], x='pulse', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='COPD?', shadow=True)

plt.title('Patients Pulse Distribution vs COPD', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);


#_______________________________second plot____________________________________
plt.subplot(2,1,2)
# ------------------------------------------board----------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50, 45]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)

#----------------------------------------------------------
df['pulse'][df.copd == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='limegreen');
df['pulse'][df.copd == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '3', fontsize=20,
                                                                color='red');

df['pulse'][df.copd == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='COPD', loc='upper right', title_fontsize=15)

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 130, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Pulse Distribution vs COPD', fontsize= 25)
plt.xlabel('Mean Pulse', fontsize= 20, labelpad=10)
plt.ylabel('Count', fontsize= 15)

plt.xlim(40,130);
plt.savefig(copd_dir+"/" + "Pulse Distribution vs COPD");
plt.savefig(pulse_dir+"/" + "Pulse Distribution vs COPD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(65, 250+2, 2)

plt.hist(data=df[df.copd == 'no'], x='sys_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.copd == 'yes'], x='sys_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.copd == 'unknown'], x='sys_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='COPD?', shadow=True)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs COPD', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(65, 250);


#==================================second plot===============================
plt.subplot(2,1,2)
#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.13)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.13)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.07)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------
df['sys_bp'][df.copd == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['sys_bp'][df.copd == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');

df['sys_bp'][df.copd == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='COPD', loc='upper right', title_fontsize=15)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs COPD', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(65, 250);
plt.savefig(copd_dir+"/" + "Systolic Blood Pressure Distribution vs COPD");
plt.savefig(sys_bp_dir+"/" + "Systolic Blood Pressure Distribution vs COPD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------
#___________________________________________________________________
bins = np.arange(30, 150+2, 2)

plt.hist(data=df[df.copd == 'no'], x='dia_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.copd == 'yes'], x='dia_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.copd == 'unknown'], x='dia_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='COPD?', shadow=True)

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs COPD', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);

#_______________________________second plot____________________________________
plt.subplot(2,1,2)
#_______________________________board____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.15)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.15)
#----------------------------------------------------------------------

df['dia_bp'][df.copd == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['dia_bp'][df.copd == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');
df['dia_bp'][df.copd == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');
plt.legend(['no','yes', 'unknown'], title='COPD?', loc='upper right')

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs COPD', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);
plt.savefig(copd_dir+"/" + "Diastolic Blood Pressure Distribution vs COPD");
plt.savefig(dia_bp_dir+"/" + "Diastolic Blood Pressure Distribution vs COPD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{cancer}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='cancer', hue='hypertension', palette=[(1,0,0,.7), (.3,1,0,.99), 'yellow', 'blue']);

plt.title('Cancer Distribution vs Hypertension', fontsize= 20, pad=10)
plt.xlabel('Cancer ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cancer_dir+"/" + "Cancer Distribution vs Hypertension");
plt.savefig(hypertension_dir+"/" + "Cancer Distribution vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='cancer', hue='diabetes', palette=[(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'blue']);

plt.title('Cancer Distribution vs Diabetes', fontsize= 20, pad=10)
plt.xlabel('Cancer ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(cancer_dir+"/" + "Cancer Distribution vs Diabetes");
plt.savefig(diabetes_dir+"/" + "Cancer Distribution vs Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,15])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df[df.cancer == 'no'], x='pulse', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.cancer == 'yes'], x='pulse', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.cancer == 'unknown'], x='pulse', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='Cancer?', shadow=True)

plt.title('Patients Pulse Distribution vs Cancer', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);


#_______________________________second plot____________________________________
plt.subplot(2,1,2)
# ------------------------------------------board----------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50, 45]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)

#----------------------------------------------------------
df['pulse'][df.cancer == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='limegreen');
df['pulse'][df.cancer == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '3', fontsize=20,
                                                                color='red');

df['pulse'][df.cancer == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='Cancer', loc='upper right', title_fontsize=15)

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 130, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Pulse Distribution vs Cancer', fontsize= 25)
plt.xlabel('Mean Pulse', fontsize= 20, labelpad=10)
plt.ylabel('Count', fontsize= 15)

plt.xlim(40,130);
plt.savefig(cancer_dir+"/" + "Pulse Distribution vs Cancer");
plt.savefig(pulse_dir+"/" + "Pulse Distribution vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________

bins = np.arange(65, 250+2, 2)

plt.hist(data=df[df.cancer == 'no'], x='sys_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.cancer == 'yes'], x='sys_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.cancer == 'unknown'], x='sys_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='Cancer?', shadow=True)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs Cancer', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(65, 250);


#==================================second plot===============================
plt.subplot(2,1,2)
#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.13)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.13)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.07)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------
df['sys_bp'][df.cancer == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['sys_bp'][df.cancer == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');

df['sys_bp'][df.cancer == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='Cancer', loc='upper right', title_fontsize=15)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs Cancer', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(65, 250);
plt.savefig(cancer_dir+"/" + "Systolic Blood Pressure Distribution vs Cancer");
plt.savefig(sys_bp_dir+"/" + "Systolic Blood Pressure Distribution vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------
#___________________________________________________________________
bins = np.arange(30, 150+2, 2)

plt.hist(data=df[df.cancer == 'no'], x='dia_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.cancer == 'yes'], x='dia_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.cancer == 'unknown'], x='dia_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='Cancer?', shadow=True)

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs Cancer', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);

#_______________________________second plot____________________________________
plt.subplot(2,1,2)
#_______________________________board____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.15)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.15)
#----------------------------------------------------------------------

df['dia_bp'][df.cancer == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['dia_bp'][df.cancer == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');
df['dia_bp'][df.cancer == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');
plt.legend(['no','yes', 'unknown'], title='Cancer?', loc='upper right')

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs Cancer', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);
plt.savefig(cancer_dir+"/" + "Diastolic Blood Pressure Distribution vs Cancer");
plt.savefig(dia_bp_dir+"/" + "Diastolic Blood Pressure Distribution vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{hypertension}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[10,6])
sb.countplot(data=df, x='hypertension', hue='diabetes', palette=[(1,0,0,.7), 'lightsalmon', (.3,1,0,.99), 'blue']);

plt.title('Hypertension Distribution vs Diabetes', fontsize= 20, pad=10)
plt.xlabel('Hypertension ?', fontsize= 15)
plt.ylabel('Count', fontsize= 13);
plt.savefig(hypertension_dir+"/" + "Hypertension Distribution vs Diabetes");
plt.savefig(diabetes_dir+"/" + "Hypertension Distribution vs Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,15])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df[df.hypertension == 'no'], x='pulse', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.hypertension == 'yes'], x='pulse', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.hypertension == 'unknown'], x='pulse', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'unknown'], loc='upper right', title='Hypertension?', shadow=True)

plt.title('Patients Pulse Distribution vs Hypertension', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);


#_______________________________second plot____________________________________
plt.subplot(2,1,2)
# ------------------------------------------board----------------------
_ls1 = [100, 105, 110, 115, 120, 130]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=130, color='r', alpha=.05)

_ls2 = [60, 55, 50, 45]
for i in _ls2:
    plt.axvspan(xmin=40,xmax=i, color='r', alpha=.05)

#----------------------------------------------------------
df['pulse'][df.hypertension == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=20, 
                                                                    color='limegreen');
df['pulse'][df.hypertension == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '3', fontsize=20,
                                                                color='red');

df['pulse'][df.hypertension == 'unknown'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=20,
                                                                color='blue');

plt.legend(['no','yes', 'unknown'], title='Hypertension', loc='upper right', title_fontsize=15)

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 130, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Pulse Distribution vs Hypertension', fontsize= 25)
plt.xlabel('Mean Pulse', fontsize= 20, labelpad=10)
plt.ylabel('Count', fontsize= 15)

plt.xlim(40,130);
plt.savefig(hypertension_dir+"/" + "Pulse Distribution vs Hypertension");
plt.savefig(pulse_dir+"/" + "Pulse Distribution vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
# ------------------------------------------board----------------------

#_______________________________first plot____________________________________
#plt.subplot(1,2,1)
bins = np.arange(65, 250+2, 2)

plt.hist(data=df[df.hypertension == 'no'], x='sys_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.hypertension == 'yes'], x='sys_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.hypertension == 'hypotension'], x='sys_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'hypotension'], loc='upper right', title='Hypertension?', shadow=True)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs Hypertension', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(65, 250);


#==================================second plot===============================
plt.subplot(2,1,2)
#_______________________________board____________________________________
# green background
plt.axvspan(xmin=90,xmax=120, color='g', alpha=.3)

# red background
_ls1 = [180, 200, 220, 240]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=250, color='r', alpha=.13)

_ls2 = [90, 80, 70]
for i in _ls2:
    plt.axvspan(xmin=65,xmax=i, color='r', alpha=.13)

# yellow background
plt.axvspan(xmin=120,xmax=130, color='yellow', alpha=.3)

# orange background
plt.axvspan(xmin=130,xmax=140, color='gold', alpha=.5)

# red background
plt.axvspan(xmin=140,xmax=180, color='r', alpha=.07)

sb.set_theme(style='whitegrid')
#----------------------------------------------------------------------
df['sys_bp'][df.hypertension == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['sys_bp'][df.hypertension == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');

df['sys_bp'][df.hypertension == 'hypotension'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');

plt.legend(['no','yes', 'hypotension'], title='Hypertensio', loc='upper right', title_fontsize=15)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='w', alpha=.25)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution vs Hypertension', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 20, labelpad=15)
plt.ylabel('Count', fontsize= 15)

plt.xlim(65, 250);
plt.savefig(hypertension_dir+"/" + "Systolic Blood Pressure Distribution vs Hypertension");
plt.savefig(sys_bp_dir+"/" + "Systolic Blood Pressure Distribution vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,20])
sb.set_theme(style='whitegrid')

#_______________________________first plot____________________________________
plt.subplot(2,1,1)
# ------------------------------------------board----------------------
#___________________________________________________________________
bins = np.arange(30, 150+2, 2)

plt.hist(data=df[df.hypertension == 'no'], x='dia_bp', bins= bins, color='limegreen', alpha=.99);
plt.hist(data=df[df.hypertension == 'yes'], x='dia_bp', bins= bins, color='red', alpha=.99);
plt.hist(data=df[df.hypertension == 'hypotension'], x='dia_bp', bins= bins, color="#00AFBB", alpha=.5);
plt.legend(['no', 'yes', 'hypotension'], loc='upper right', title='Hypertension?', shadow=True)

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs Hypertension', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);

#_______________________________second plot____________________________________
plt.subplot(2,1,2)
#_______________________________board____________________________________
plt.axvspan(xmin=60,xmax=80, color='g', alpha=.1)
plt.axvspan(xmin=80,xmax=90, color='orange', alpha=.1)
plt.axvspan(xmin=90,xmax=120, color='r', alpha=.1)

_ls1 = [120, 130, 140]
for i in _ls1:
    plt.axvspan(xmin=i,xmax=150, color='r', alpha=.15)

_ls2 = [60, 50, 40]
for i in _ls2:
    plt.axvspan(xmin=30,xmax=i, color='r', alpha=.15)
#----------------------------------------------------------------------

df['dia_bp'][df.hypertension == 'no'].value_counts().sort_index().plot(kind='line', 
                                                                    linewidth = '2', fontsize=15, 
                                                                    color='limegreen');
df['dia_bp'][df.hypertension == 'yes'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='red');
df['dia_bp'][df.hypertension == 'hypotension'].value_counts().sort_index().plot(kind='line',
                                                                  linewidth = '2', fontsize=15,
                                                                color='blue');
plt.legend(['no','yes', 'hypotension'], title='Hypertension?', loc='upper right')

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Diastolic Blood Pressure Distribution vs Hypertension', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);
plt.savefig(hypertension_dir+"/" + "Diastolic Blood Pressure Distribution vs Hypertension");
plt.savefig(dia_bp_dir+"/" + "Diastolic Blood Pressure Distribution vs Hypertension");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{pulse, sys_bp, dia_bp}
diseases_features = ['pulse', 'sys_bp', 'dia_bp']
g = sb.PairGrid(data=df, vars=diseases_features)
g.map_offdiag(plt.scatter)
g.map_diag(plt.hist);
plt.savefig(pulse_dir+"/" + "pair plot Pulse-systolic pressure-Diastolic pressure");
plt.savefig(sys_bp_dir+"/" + "pair plot Pulse-systolic pressure-Diastolic pressure");
plt.savefig(dia_bp_dir+"/" + "pair plot Pulse-systolic pressure-Diastolic pressure");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$
diseases_features = ['pulse', 'sys_bp', 'dia_bp']
correlations = df[diseases_features].corr()

plt.figure(figsize= [10,6])
sb.heatmap(correlations, cmap='PRGn', annot=True, fmt='.2f', center = 0)

plt.title('Diseases Data Correlations  (Paired)', fontsize= 22, pad=15)
plt.xlabel('Features', fontsize= 15, labelpad=20)
plt.ylabel('Features', fontsize= 15, labelpad=20);
plt.savefig(pulse_dir+"/" + "Correlations Pulse-systolic pressure-Diastolic pressure");
plt.savefig(sys_bp_dir+"/" + "Correlations Pulse-systolic pressure-Diastolic pressure");
plt.savefig(dia_bp_dir+"/" + "Correlations Pulse-systolic pressure-Diastolic pressure");
#$$$$$$$$$$$$$$$$$$$$$$$$$$
#			{Multivariate Diseases Analytics}
#	{chf, mi, cad}heart diseases
plt.figure(figsize=[20, 8])
sb.set_theme(style='whitegrid')

plt.subplot(1,3,1)
sb.countplot(data=df[(df.cad == 'yes')], x='chf', order=['yes', 'no', 'unknown'], 
             hue='mi', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("Coronary artery disease CAD = 'YES'", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('CHF', labelpad=10);

# --------------------------------------------------
plt.subplot(1,3,2)
sb.countplot(data=df[(df.cad == 'no')], x='chf', order=['yes', 'no', 'unknown'], 
             hue='mi', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("Coronary artery disease CAD = 'NO'", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('CHF', labelpad=10);

# --------------------------------------------------
#plt.subplot(1,3,3)
#sb.countplot(data=df[(df.cad == 'unknown')], x='chf', order=['yes', 'no', 'unknown'], 
#             hue='mi', palette=['red', 'green', 'blue'])

#plt.xticks(rotation=0)
#plt.title("Coronary artery disease CAD = 'unknown'", fontsize= 15)
#plt.ylabel('Count', labelpad=10)
#plt.xlabel('CHF', labelpad=10);
plt.savefig(chf_dir+"/" + "CHF vs MI vs CAD");
plt.savefig(mi_dir+"/" + "CHF vs MI vs CAD");
plt.savefig(cad_dir+"/" + "CHF vs MI vs CAD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{asthma, copd'emphysema', cancer}
plt.figure(figsize=[20, 8])
sb.set_theme(style='whitegrid')

plt.subplot(1,3,1)
sb.countplot(data=df[(df.copd == 'yes')], x='asthma', order=['yes', 'no', 'unknown'], 
             hue='cancer', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("chronic obstructive pulmonary disease (COPD) = 'YES'", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Asthma', labelpad=10);

# --------------------------------------------------
plt.subplot(1,3,2)
sb.countplot(data=df[(df.cad == 'no')], x='asthma', order=['yes', 'no', 'unknown'], 
             hue='cancer', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("chronic obstructive pulmonary disease (COPD) = 'NO'", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Asthma', labelpad=10);

# --------------------------------------------------
#plt.subplot(1,3,3)
#sb.countplot(data=df[(df.cad == 'unknown')], x='asthma', order=['yes', 'no', 'unknown'], 
#             hue='cancer', palette=['red', 'green', 'blue'])
#
#plt.xticks(rotation=0)
#plt.title("chronic obstructive pulmonary disease (COPD) = 'unknown'", fontsize= 15)
#plt.ylabel('Count', labelpad=10)
#plt.xlabel('Asthma', labelpad=10);
#plt.savefig(asthma_dir+"/" + "Asthma vs COPD vs Cancer");
#plt.savefig(copd_dir+"/" + "Asthma vs COPD vs Cancer");
#plt.savefig(cancer_dir+"/" + "Asthma vs COPD vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{strok(cva), hypertension, diabetes}
plt.figure(figsize=[20, 15])
sb.set_theme(style='whitegrid')

plt.subplot(2,2,1)
sb.countplot(data=df[(df.diabetes == 'yes')], x='hypertension', order=['yes', 'no', 'hypotension', 'unknown'], 
             hue='cva', palette=['red', 'green', 'blue'])
plt.legend(loc='upper right', title='cva')
plt.xticks(rotation=0)
plt.title("Diabetic patients? 'YES'", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Hypertension', labelpad=10);

# --------------------------------------------------
plt.subplot(2,2,2)
sb.countplot(data=df[(df.diabetes == 'borderline')], x='hypertension', order=['yes', 'no', 'hypotension', 'unknown'], 
             hue='cva', palette=['red', 'green', 'blue'])
plt.legend(loc='upper right', title='cva')
plt.xticks(rotation=0)
plt.title("Diabetic patients? 'Borderline'", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Hypertension', labelpad=10);

# --------------------------------------------------
plt.subplot(2,2,3)
sb.countplot(data=df[(df.diabetes == 'no')], x='hypertension', order=['yes', 'no', 'hypotension', 'unknown'], 
             hue='cva', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("Diabetic patients? 'NO'", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Hypertension', labelpad=10);

# --------------------------------------------------
#plt.subplot(2,2,4)
#sb.countplot(data=df[(df.diabetes == 'unknown')], x='hypertension', order=['yes', 'no', 'hypotension', 'unknown'], 
#             hue='cva', palette=['red', 'green', 'blue'])
#
#plt.xticks(rotation=0)
#plt.title("Diabetic patients? 'unknown'", fontsize= 15)
#plt.ylabel('Count', labelpad=10)
#plt.xlabel('Hypertension', labelpad=10);
plt.savefig(cva_dir+"/" + "Asthma vs COPD vs Cancer");
plt.savefig(hypertension_dir+"/" + "Asthma vs COPD vs Cancer");
plt.savefig(diabetes_dir+"/" + "Asthma vs COPD vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#						{{{{{{{{{{{Diseases and personal data Analytics}}}}}}}}}}}

#	{diseases by gender}
plt.figure(figsize=[10, 6])

ax = sb.countplot(data=df, x='asthma', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'])

plt.xticks(rotation=0)
plt.title("Asthma patients Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Asthma patients?', labelpad=20);
plt.savefig(d_gender_dir+"/" + "Asthma patients Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[10, 6])

sb.countplot(data=df, x='chf', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'])

plt.xticks(rotation=0)
plt.title("CHF patients Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('CHF patients?', labelpad=20);
plt.savefig(d_gender_dir+"/" + "CHF patients Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[10, 6])

sb.countplot(data=df, x='cad', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'])

plt.xticks(rotation=0)
plt.title("CAD patients Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('CAD patients?', labelpad=20);
plt.savefig(d_gender_dir+"/" + "CAD patients Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[10, 6])

sb.countplot(data=df, x='mi', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'])

plt.xticks(rotation=0)
plt.title("MI patients Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('MI patients?', labelpad=20);
plt.savefig(d_gender_dir+"/" + "MI patients Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[10, 6])

sb.countplot(data=df, x='cva', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'])

plt.xticks(rotation=0)
plt.title("Stroke patients Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('CVA patients?', labelpad=20);
plt.savefig(d_gender_dir+"/" + "Stroke patients Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[10, 6])

sb.countplot(data=df, x='copd', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'])

plt.xticks(rotation=0)
plt.title("COPD patients Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('COPD patients?', labelpad=20);
plt.savefig(d_gender_dir+"/" + "COPD patients Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[10, 6])

sb.countplot(data=df, x='cancer', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'])

plt.xticks(rotation=0)
plt.title("Cancer patients Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Cancer patients?', labelpad=20);
plt.savefig(d_gender_dir+"/" + "Cancer patients Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[10, 6])

sb.countplot(data=df, x='hypertension', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'])

plt.xticks(rotation=0)
plt.title("Hypertension patients Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Hypertension patients?', labelpad=20);
plt.savefig(d_gender_dir+"/" + "Hypertension patients Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[10, 6])

sb.countplot(data=df, x='diabetes', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'])

plt.xticks(rotation=0)
plt.title("Diabetic patients Distribution for both genders", fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Diabetic patients?', labelpad=20);
plt.savefig(d_gender_dir+"/" + "Diabetic patients Distribution for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
bins = np.arange(40, 130+2, 2)

plt.hist(data=df[df.gender == 'female'], x='pulse', bins= bins, color='pink', alpha=.99);
plt.hist(data=df[df.gender == 'male'], x='pulse', bins= bins, color='blue', alpha=.25);
plt.legend(['female', 'male'], loc='upper right', title='Gender', shadow=True)

plt.title('Patients Pulse Distribution by Gender', fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Mean Pulse', labelpad=10);
#plt.axvline(x=df.pulse.mean(), linestyle='--', linewidth=2, color='b')
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

ticks = np.arange(40, 140, 10)
ticks = list(ticks)
#ticks.append(df.pulse.mean())
ticks.append(60)
ticks.append(100)
ticks= [int(x) for x in ticks]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.xlim(40, 130);


plt.subplot(2,1,2)
sb.violinplot(data=df, x='pulse', y='gender', order=['female', 'male'], orient='horizontal', palette=['pink', 'cornflowerblue']);

plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=100, linestyle='-', linewidth=2, color='r', alpha=.75)

plt.title('Patients Pulse Distribution by Gender', fontsize= 20)
plt.xlabel('Heart rate', labelpad=10)
plt.ylabel('Gender', labelpad=10);
plt.savefig(d_gender_dir+"/" + "Patients Pulse Distribution by Gender");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)

bins = np.arange(65, 250+2, 2)

plt.hist(data=df[df.gender == 'female'], x='sys_bp', bins= bins, color='pink', alpha=.99);
plt.hist(data=df[df.gender == 'male'], x='sys_bp', bins= bins, color='blue', alpha=.25);
plt.legend(['female', 'male'], loc='upper right', title='Gender', shadow=True)

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution by Gender', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 15, labelpad=15)
plt.ylabel('Count')
plt.xlim(65, 250);


plt.subplot(2,1,2)
sb.violinplot(data=df, x='sys_bp', y='gender', order=['female', 'male'], orient='horizontal', palette=['pink', 'cornflowerblue']);

plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axvline(x=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axvline(x=180, linestyle='-', linewidth=2, color='r', alpha=1)

ticks = np.arange(60, 260, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Systolic Blood Pressure Distribution by Gender', fontsize= 25)
plt.xlabel('Mean Systolic BP', fontsize= 15, labelpad=15)
plt.ylabel('Gender')
plt.xlim(65, 250);
plt.savefig(d_gender_dir+"/" + "Systolic Blood Pressure Distribution by Gender");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)

bins = np.arange(30, 150+2, 2)

plt.hist(data=df[df.gender == 'female'], x='dia_bp', bins= bins, color='pink', alpha=.99);
plt.hist(data=df[df.gender == 'male'], x='dia_bp', bins= bins, color='blue', alpha=.25);
plt.legend(['female', 'male'], loc='upper right', title='Gender', shadow=True)

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=0)
plt.title('Diastolic Blood Pressure Distribution by Gender', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Count', fontsize= 15)
plt.xlim(30, 150);


plt.subplot(2,1,2)
sb.violinplot(data=df, x='dia_bp', y='gender', order=['female', 'male'], orient='horizontal', palette=['pink', 'cornflowerblue']);

plt.axvline(x=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axvline(x=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axvline(x=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axvline(x=90, linestyle='-', linewidth=2, color='r', alpha=.25)

ticks = np.arange(30, 160, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=0)
plt.title('Diastolic Blood Pressure Distribution by Gender', fontsize= 22)
plt.xlabel('Mean Diastolic BP', fontsize= 18, labelpad=15)
plt.ylabel('Gender', fontsize= 15)
plt.xlim(30, 150);
plt.savefig(d_gender_dir+"/" + "Diastolic Blood Pressure Distribution by Gender");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{diseases by age}
plt.figure(figsize=[15, 15])

plt.subplot(2,1,1)
#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
plt.hist(data=df[df.asthma == 'yes'], x='age', bins= bins, color='red', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.asthma == 'no'], x='age', bins= bins, color='green', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.asthma == 'unknown'], x='age', bins= bins, color='blue', alpha=.99, histtype='step', lw=3);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.legend(['yes', 'no', 'unknown'], loc='upper right', title='Asthma patient', shadow=True)
plt.xticks(ticks, labels)

#plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution For Asthma patients', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);

#--------------------------------------------
plt.subplot(2,1,2)
sb.violinplot(data=df, x='asthma', y='age', color='deepskyblue', inner='quartile');
plt.savefig(d_age_dir+"/" + "Age Distribution For Asthma patients");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 15])

plt.subplot(2,1,1)
#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
plt.hist(data=df[df.chf == 'yes'], x='age', bins= bins, color='red', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.chf == 'no'], x='age', bins= bins, color='green', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.chf == 'unknown'], x='age', bins= bins, color='blue', alpha=.99, histtype='step', lw=3);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.legend(['yes', 'no', 'unknown'], loc='upper right', title='CHF patient', shadow=True)
plt.xticks(ticks, labels)

#plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution For CHF patients', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);

#--------------------------------------------
plt.subplot(2,1,2)
sb.violinplot(data=df, x='chf', y='age', color='deepskyblue', inner='quartile');
plt.savefig(d_age_dir+"/" + "Age Distribution For CHF patients");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 15])

plt.subplot(2,1,1)

#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
plt.hist(data=df[df.cad == 'yes'], x='age', bins= bins, color='red', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.cad == 'no'], x='age', bins= bins, color='green', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.cad == 'unknown'], x='age', bins= bins, color='blue', alpha=.99, histtype='step', lw=3);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.legend(['yes', 'no', 'unknown'], loc='upper right', title='CAD patient', shadow=True)
plt.xticks(ticks, labels)

#plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution For CAD patients', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);

#--------------------------------------------
plt.subplot(2,1,2)
sb.violinplot(data=df, x='cad', y='age', color='deepskyblue', inner='quartile');
plt.savefig(d_age_dir+"/" + "Age Distribution For CAD patients");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 15])

plt.subplot(2,1,1)

#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
plt.hist(data=df[df.mi == 'yes'], x='age', bins= bins, color='red', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.mi == 'no'], x='age', bins= bins, color='green', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.mi == 'unknown'], x='age', bins= bins, color='blue', alpha=.99, histtype='step', lw=3);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.legend(['yes', 'no', 'unknown'], loc='upper right', title='MI patient', shadow=True)
plt.xticks(ticks, labels)

#plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution For MI patients', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);

#--------------------------------------------
plt.subplot(2,1,2)
sb.violinplot(data=df, x='mi', y='age', color='deepskyblue', inner='quartile');
plt.savefig(d_age_dir+"/" + "Age Distribution For MI patients");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 15])

plt.subplot(2,1,1)

#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
plt.hist(data=df[df.cva == 'yes'], x='age', bins= bins, color='red', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.cva == 'no'], x='age', bins= bins, color='green', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.cva == 'unknown'], x='age', bins= bins, color='blue', alpha=.99, histtype='step', lw=3);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.legend(['yes', 'no', 'unknown'], loc='upper right', title='CVA patient', shadow=True)
plt.xticks(ticks, labels)

#plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution For CVA patients', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);

#-----------------------------------------
plt.subplot(2,1,2)
sb.violinplot(data=df, x='cva', y='age', color='deepskyblue', inner='quartile');
plt.savefig(d_age_dir+"/" + "Age Distribution For CVA patients");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 15])

plt.subplot(2,1,1)
#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
plt.hist(data=df[df.copd == 'yes'], x='age', bins= bins, color='red', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.copd == 'no'], x='age', bins= bins, color='green', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.copd == 'unknown'], x='age', bins= bins, color='blue', alpha=.99, histtype='step', lw=3);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.legend(['yes', 'no', 'unknown'], loc='upper right', title='COPD patient', shadow=True)
plt.xticks(ticks, labels)

#plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution For COPD patients', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);

#-------------------------------------------
plt.subplot(2,1,2)
sb.violinplot(data=df, x='copd', y='age', color='deepskyblue', inner='quartile');
plt.savefig(d_age_dir+"/" + "Age Distribution For COPD patients");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 15])

plt.subplot(2,1,1)
#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
plt.hist(data=df[df.cancer == 'yes'], x='age', bins= bins, color='red', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.cancer == 'no'], x='age', bins= bins, color='green', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.cancer == 'unknown'], x='age', bins= bins, color='blue', alpha=.99, histtype='step', lw=3);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.legend(['yes', 'no', 'unknown'], loc='upper right', title='Cancer patient', shadow=True)
plt.xticks(ticks, labels)

#plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution For Cancer patients', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);

#------------------------------------------------
plt.subplot(2,1,2)
sb.violinplot(data=df, x='cancer', y='age', color='deepskyblue', inner='quartile');
plt.savefig(d_age_dir+"/" + "Age Distribution For Cancer patients");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 15])

plt.subplot(2,1,1)
#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
plt.hist(data=df[df.hypertension == 'yes'], x='age', bins= bins, color='red', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.hypertension == 'no'], x='age', bins= bins, color='green', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.hypertension == 'hypotension'], x='age', bins= bins, color='yellow', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.hypertension == 'unknown'], x='age', bins= bins, color='blue', alpha=.99, histtype='step', lw=3);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.legend(['yes', 'no', 'hypotension', 'unknown'], loc='upper right', title='Hypertension patient', shadow=True)
plt.xticks(ticks, labels)

#plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution For Hypertension patients', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);

#----------------------------------------
plt.subplot(2,1,2)
sb.violinplot(data=df, x='hypertension', y='age', color='deepskyblue', inner='quartile');
plt.savefig(d_age_dir+"/" + "Age Distribution For Hypertension patients");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15, 15])

plt.subplot(2,1,1)
#_______________________________first plot____________________________________
bins = np.arange(0, df.age.max()+3, 3)
plt.hist(data=df[df.diabetes == 'yes'], x='age', bins= bins, color='red', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.diabetes == 'no'], x='age', bins= bins, color='green', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.diabetes == 'borderline'], x='age', bins= bins, color='lightsalmon', alpha=.99, histtype='step', lw=3);
plt.hist(data=df[df.diabetes == 'unknown'], x='age', bins= bins, color='blue', alpha=.99, histtype='step', lw=3);

ticks = np.arange(0, 110, 10)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.legend(['yes', 'no', 'borderline', 'unknown'], loc='upper right', title='Diabetic patient', shadow=True)
plt.xticks(ticks, labels)

#plt.axvline(x=df.age.mean(), linestyle='--', linewidth=2, color='r')

plt.title('Age Distribution For Diabetic patients', fontsize= 15)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Age', labelpad=10);

#----------------------------------------------------
plt.subplot(2,1,2)
sb.violinplot(data=df, x='diabetes', y='age', color='deepskyblue', inner='quartile');
plt.savefig(d_age_dir+"/" + "Age Distribution For Diabetic patients");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15,7])
sb.regplot(data=df, x='age', y='pulse', x_jitter=0.2, scatter_kws={'alpha':.7})
plt.title('Age Distribution vs patients Heart rate', fontsize=20);
plt.savefig(d_age_dir+"/" + "Age Distribution vs patients Heart rate");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15,7])
sb.regplot(data=df, x='age', y='sys_bp', x_jitter=0.2, scatter_kws={'alpha':.2}, truncate=False)
plt.title('Age Distribution vs patients Systolic BP', fontsize=20);
plt.savefig(d_age_dir+"/" + "Age Distribution vs patients Systolic BP");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15,7])
sb.regplot(data=df, x='age', y='sys_bp', x_jitter=0.2, scatter_kws={'alpha':.2}, truncate=False)
plt.title('Age Distribution vs patients Diastolic BP', fontsize=20);
plt.savefig(d_age_dir+"/" + "Age Distribution vs patients Diastolic BP");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
some_features = ['age', 'sys_bp', 'dia_bp']
correlations = df[some_features].corr()

plt.figure(figsize= [10,5])
sb.heatmap(correlations, cmap='BrBG', annot=True, fmt='.2f', center = 0)

plt.title('Age - Blood Pressures Correlations  (Paired)', fontsize= 15, pad=15)
plt.xlabel('', fontsize= 15, labelpad=20)
plt.ylabel('', fontsize= 15, labelpad=20);
plt.savefig(d_age_dir+"/" + "Age - Blood Pressures Correlations  (Paired)");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{Diseases vs general health}
plt.figure(figsize=[14, 7])

ax = sb.countplot(data=df, x='gen_health', hue='asthma', 
             palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("Asthma patients Distribution By General Health Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('General Health Condition', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "Asthma patients Distribution By General Health Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 7])

ax = sb.countplot(data=df, x='gen_health', hue='chf', 
             palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("CHF patients Distribution By General Health Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('General Health Condition', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "CHF patients Distribution By General Health Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 7])

ax = sb.countplot(data=df, x='gen_health', hue='cad', 
             palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("CAD patients Distribution By General Health Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('General Health Condition', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "CAD patients Distribution By General Health Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 7])

ax = sb.countplot(data=df, x='gen_health', hue='mi', 
             palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("MI patients Distribution By General Health Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('General Health Condition', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "MI patients Distribution By General Health Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 7])

ax = sb.countplot(data=df, x='gen_health', hue='cva', 
             palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("CVA patients Distribution By General Health Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('General Health Condition', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "CVA patients Distribution By General Health Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 7])

ax = sb.countplot(data=df, x='gen_health', hue='copd', 
             palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("COPD patients Distribution By General Health Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('General Health Condition', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "COPD patients Distribution By General Health Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 7])

ax = sb.countplot(data=df, x='gen_health', hue='cancer', 
             palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("Cancer patients Distribution By General Health Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('General Health Condition', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "Cancer patients Distribution By General Health Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 7])

ax = sb.countplot(data=df, x='gen_health', hue='hypertension', 
             palette=['red', 'green', 'yellow', 'blue'])

plt.xticks(rotation=0)
plt.title("Hypertension patients Distribution By General Health Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('General Health Condition', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "Hypertension patients Distribution By General Health Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 7])

ax = sb.countplot(data=df, x='gen_health', hue='diabetes', 
             palette=['red', 'lightsalmon', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("Diabetes patients Distribution By General Health Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('General Health Condition', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "Diabetes patients Distribution By General Health Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[12,10])
sb.set_theme(style='whitegrid')

plt.axhline(y=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axhline(y=100, linestyle='-', linewidth=2, color='r', alpha=.75)

sb.violinplot(data=df, x='gen_health', y='pulse', color=sb.color_palette('rainbow', 10)[0])

plt.title('Patients Pulse Distribution vs General Health', fontsize=20, pad=20)
plt.xlabel('General Health Conditions', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "Patients Pulse Distribution vs General Health");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[12,10])
sb.set_theme(style='white')

plt.axhline(y=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axhline(y=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axhline(y=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axhline(y=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axhline(y=180, linestyle='-', linewidth=2, color='r', alpha=1)

sb.violinplot(data=df, x='gen_health', y='sys_bp', color=sb.color_palette('rainbow', 10)[2])

plt.title('Patients Systolic BP Distribution vs General Health', fontsize=20)
plt.xlabel('General Health Conditions', fontsize=15, labelpad=10);
plt.savefig(d_genera_health_dir+"/" + "Patients Systolic BP Distribution vs General Health");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[12,10])
sb.set_theme(style='white')

plt.axhline(y=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axhline(y=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axhline(y=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axhline(y=90, linestyle='-', linewidth=2, color='r', alpha=.25)

sb.violinplot(data=df, x='gen_health', y='dia_bp', color=sb.color_palette('rainbow', 10)[3])

plt.title('Patients Diastolic Distribution vs General Health', fontsize=20, pad=20)
plt.xlabel('General Health Conditions', fontsize=15, labelpad=20);
plt.savefig(d_genera_health_dir+"/" + "Patients Diastolic BP Distribution vs General Health");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{asthma, emphysema(copd), and smoking}
plt.figure(figsize=[14, 8])
sb.set_theme(style='whitegrid')

sb.countplot(data=df, x='smoker', hue='asthma', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("Asthma patients Distribution vs Smoking Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Smoking Condition', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "Asthma patients Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 8])
sb.set_theme(style='whitegrid')

sb.countplot(data=df, x='smoker', hue='copd', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("COPD patients Distribution vs Smoking Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Smoking Condition', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "COPD patients Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 8])
sb.set_theme(style='whitegrid')

sb.countplot(data=df, x='smoker', hue='chf', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("CHF patients Distribution vs Smoking Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Smoking Condition', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "CHF patients Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 8])
sb.set_theme(style='whitegrid')

sb.countplot(data=df, x='smoker', hue='cad', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("CAD patients Distribution vs Smoking Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Smoking Condition', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "CAD patients Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 8])
sb.set_theme(style='whitegrid')

sb.countplot(data=df, x='smoker', hue='mi', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("MI patients Distribution vs Smoking Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Smoking Condition', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "MI patients Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 8])
sb.set_theme(style='whitegrid')

sb.countplot(data=df, x='smoker', hue='cva', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("CVA patients Distribution vs Smoking Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Smoking Condition', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "CVA patients Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 8])
sb.set_theme(style='whitegrid')

sb.countplot(data=df, x='smoker', hue='cancer', palette=['red', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("Cancer patients Distribution vs Smoking Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Smoking Condition', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "Cancer patients Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 8])
sb.set_theme(style='whitegrid')

sb.countplot(data=df, x='smoker', hue='hypertension', palette=['red', 'green', 'yellow', 'blue'])

plt.xticks(rotation=0)
plt.title("Hypertension patients Distribution vs Smoking Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Smoking Condition', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "Hypertension patients Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[14, 8])
sb.set_theme(style='whitegrid')

sb.countplot(data=df, x='smoker', hue='diabetes', palette=['red', 'lightsalmon', 'green', 'blue'])

plt.xticks(rotation=0)
plt.title("Diabetic patients Distribution vs Smoking Condition", fontsize= 20)
plt.ylabel('Count', labelpad=10)
plt.xlabel('Smoking Condition', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "Diabetic patients Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[12,10])
sb.set_theme(style='whitegrid')

plt.axhline(y=60, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axhline(y=100, linestyle='-', linewidth=2, color='r', alpha=.75)

sb.violinplot(data=df, x='smoker', y='pulse', color=sb.color_palette('Wistia', 10)[0])

plt.title('Patients Pulse Distribution vs Smoking Condition', fontsize=20, pad=20)
plt.xlabel('Smoking Conditions', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "Patients Pulse Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[12,10])
sb.set_theme(style='white')

plt.axhline(y=90, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axhline(y=120, linestyle='-', linewidth=2, color='y', alpha=.5)
plt.axhline(y=130, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axhline(y=140, linestyle='-', linewidth=2, color='r', alpha=.75)
plt.axhline(y=180, linestyle='-', linewidth=2, color='r', alpha=1)

sb.violinplot(data=df, x='smoker', y='sys_bp', color=sb.color_palette('Wistia', 10)[2])

plt.title('Patients Systolic BP Distribution vs Smoking Condition', fontsize=20)
plt.xlabel('Smoking Conditions', fontsize=15, labelpad=10);
plt.savefig(d_smoker_dir+"/" + "Patients Systolic BP Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[12,10])
sb.set_theme(style='white')

plt.axhline(y=120, linestyle='-', linewidth=2, color='r', alpha=1)
plt.axhline(y=60, linestyle='-', linewidth=2, color='r', alpha=.5)
plt.axhline(y=80, linestyle='-', linewidth=2, color='r', alpha=.25)
plt.axhline(y=90, linestyle='-', linewidth=2, color='r', alpha=.25)

sb.violinplot(data=df, x='smoker', y='dia_bp', color=sb.color_palette('Wistia', 10)[4])

plt.title('Patients Diastolic Distribution vs Smoking Condition', fontsize=20, pad=20)
plt.xlabel('General Health Conditions', fontsize=15, labelpad=20);
plt.savefig(d_smoker_dir+"/" + "Patients Systolic BP Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{chf, cad, mi, diabetes, and waist circumference}
plt.figure(figsize=[12,10])
sb.set_theme(style='whitegrid')

sb.violinplot(data=df, x='chf', y='waist_cm', color=sb.color_palette('cool', 10)[2])

plt.title('Patients Waist Circumference Distribution vs CHF', fontsize=20, pad=20)
plt.xlabel('Congestive Heart Failure', fontsize=15, labelpad=20);
plt.savefig(d_waist_dir+"/" + "Patients Waist Circumference Distribution vs CHF");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[12,10])
sb.set_theme(style='whitegrid')

sb.violinplot(data=df, x='cad', y='waist_cm', color=sb.color_palette('cool', 10)[4])

plt.title('Patients Waist Circumference Distribution vs CAD', fontsize=20, pad=20)
plt.xlabel('Coronary artery disease (CAD)', fontsize=15, labelpad=20);
plt.savefig(d_waist_dir+"/" + "Patients Waist Circumference Distribution vs CAD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[12,10])
sb.set_theme(style='whitegrid')

sb.violinplot(data=df, x='mi', y='waist_cm', color=sb.color_palette('cool', 10)[6])

plt.title('Patients Waist Circumference Distribution vs MI', fontsize=20, pad=20)
plt.xlabel('Myocardial infarction (MI)', fontsize=15, labelpad=20);
plt.savefig(d_waist_dir+"/" + "Patients Waist Circumference Distribution vs MI");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[12,10])
sb.set_theme(style='whitegrid')

sb.violinplot(data=df, x='diabetes', y='waist_cm', color=sb.color_palette('autumn_r', 10)[0])

plt.title('Patients Waist Circumference Distribution vs Diabetes', fontsize=20, pad=20)
plt.xlabel('Diabetes Conditions', fontsize=15, labelpad=20);
plt.savefig(d_waist_dir+"/" + "Patients Waist Circumference Distribution vs Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{weight and blood pressure}
plt.figure(figsize=[14, 8])
plt.scatter(data=df, x='sys_bp', y='dia_bp', c='bmi', cmap='autumn_r', alpha=1)
plt.colorbar(label='BMI (kg/m^2)');

plt.title('Blood Pressure vs Body-Mass-Index', fontsize= 15)
plt.ylabel('Diastolic Blood Pressure', labelpad=10)
plt.xlabel('Systolic Blood Pressure', labelpad=10);
plt.savefig(d_bmi_dir+"/" + "Blood Pressures vs Body-Mass-Index");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{bmi and hypertension for both genders}
plt.figure(figsize=[16, 10])
sb.set_theme(style='whitegrid')

sb.violinplot(data=df, x='hypertension', y='bmi', hue='gender', hue_order=['female', 'male'], 
             palette=['pink', 'blue'], alpha=.5)

plt.title('BMI vs Hypertension Condition for both genders', fontsize= 20, pad=15)
plt.xlabel('Hypertension Condition', fontsize= 15, labelpad=20)
plt.ylabel('Body Mass Index', fontsize= 15, labelpad=10);
plt.savefig(d_bmi_dir+"/" + "BMI vs Hypertension Condition for both genders");
plt.savefig(d_gender_dir+"/" + "BMI vs Hypertension Condition for both genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#						{{{{{{{{{{{{{{{{{{{{{{Medical Tests Analysis}}}}}}}}}}}}}}}}}}}}}}

#				{{{{1-Complete Blood Count (CBC)}}}}
#	{wbc white blood cells}
plt.figure(figsize=[20,15])
sb.set_theme(style=None)

plt.subplot(2,1,1)
bins = np.arange(1, df.wbc.max()+.2, .2)
plt.hist(data=df, x='wbc', bins= bins, color='firebrick')

plt.axvline(x= 4.5, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 11, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(0, df.wbc.max()+5, 5)
ticks = np.append(ticks, [1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19])
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('White Blood Count Distribution', pad=10, fontsize=20)
plt.xlabel('White Blood Count (billion/liter)', labelpad=10, fontsize=15)
plt.ylabel('Count');


plt.subplot(2,1,2)

sb.boxplot(data=df, x='wbc', color='r')

plt.axvline(x=4.5, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x=11, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(0, df.wbc.max()+5, 5)
ticks = np.append(ticks, [1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19])
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('White Blood Count Distribution', pad=10, fontsize=20)
plt.xlabel('White Blood Count (billion/liter)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.savefig(cbc_dir+"/" + "White Blood Count Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,6])
sb.set_theme(style=None)

sb.violinplot(data=df, x='wbc', y='gender', order=['female', 'male'], orient='horizontal', palette=['pink', 'cornflowerblue']);

plt.axvline(x=4.5, linestyle='-', linewidth=2, color='yellow', alpha=1)
plt.axvline(x=11, linestyle='-', linewidth=2, color='yellow', alpha=1)

ticks = np.arange(0, df.wbc.max()+5, 5)
ticks = np.append(ticks, [1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19])
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('White Blood Count Distribution By Gender', pad=10, fontsize=20)
plt.xlabel('White Blood Count (billion/liter)', labelpad=10, fontsize=15)
plt.ylabel('Gender')
plt.xlim(df.wbc.quantile(.01), df.wbc.quantile(.99));
plt.savefig(cbc_dir+"/" + "White Blood Count Distribution By Gender");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,8])
sb.set_theme(style=None)

sb.regplot(data=df, x='age', y='wbc', scatter_kws={'alpha':.25}, truncate=False)

plt.title('White Blood Count vs Age', fontsize=20, pad=15)
plt.xlabel('Age', fontsize=15, labelpad=10)
plt.ylabel('WBC (billion / liter)', fontsize=15, labelpad=10)

plt.ylim(df.wbc.quantile(.01), df.wbc.quantile(.99));
plt.savefig(cbc_dir+"/" + "White Blood Count vs Age");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style=None)

g = sb.FacetGrid(data=df, hue='gender', height=8, aspect=2, palette=['pink', (0,0,1,.01)])
g.map(plt.scatter, 'age', 'wbc')
g.add_legend()

plt.title('White Blood Count vs Age For Both Genders', fontsize=20, pad=15)
plt.xlabel('Age', fontsize=15, labelpad=10)
plt.ylabel('WBC (billion / liter)', fontsize=15, labelpad=10)
plt.ylim(df.wbc.quantile(.01), df.wbc.quantile(.99));
plt.savefig(cbc_dir+"/" + "White Blood Count vs Age For Both Genders");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 17])
sb.set_theme(style=None)

plt.subplot(2,1,1)
sb.regplot(data=df[df.gender == 'female'], x='age', y='wbc', scatter_kws={'alpha':.5}, truncate=False)

plt.title('White Blood Count vs Age For Females', fontsize=20, pad=15)
plt.xlabel('Age', fontsize=15, labelpad=5)
plt.ylabel('WBC (billion / liter)', fontsize=15, labelpad=10)
plt.ylim(df.wbc.quantile(.01), df.wbc.quantile(.99));


plt.subplot(2,1,2)
sb.regplot(data=df[df.gender == 'male'], x='age', y='wbc', scatter_kws={'alpha':.5}, truncate=False)           

plt.title('White Blood Count vs Age For Males', fontsize=20, pad=15)
plt.xlabel('Age', fontsize=15, labelpad=5)
plt.ylabel('WBC (billion / liter)', fontsize=15, labelpad=10)
plt.ylim(df.wbc.quantile(.01), df.wbc.quantile(.99));

plt.savefig(cbc_dir+"/" + "White Blood Count vs Age For females and males");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20, 8])
sb.set_theme(style=None)

sb.violinplot(data=df, x='wbc', y='cancer', palette=['red', 'green', 'blue'])

plt.title('White Blood Count vs Cancer condition', fontsize=20, pad=15)
plt.ylabel('Cancer Condition', fontsize=15, labelpad=5)
plt.xlabel('WBC (billion / liter)', fontsize=15, labelpad=10);
plt.xlim(df.wbc.quantile(.01), df.wbc.quantile(.99));
plt.savefig(cbc_dir+"/" + "White Blood Count vs Cancer condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{hgb}
plt.figure(figsize=[20,15])
sb.set_theme(style=None)

plt.subplot(2,1,1)
bins = np.arange(5, df.hgb.max()+.25, .25)
plt.hist(data=df, x='hgb', bins= bins, color='firebrick')

ticks = np.arange(5, df.hgb.max()+1, 1)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Hemoglobin Distribution', pad=10, fontsize=20)
plt.xlabel('Hemoglobin (g /dL)', labelpad=10, fontsize=15)
plt.ylabel('Count');


plt.subplot(2,1,2)
sb.boxplot(data=df, x='hgb', color='r')

ticks = np.arange(5, df.hgb.max()+1, 1)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Hemoglobin Distribution', pad=10, fontsize=20)
plt.xlabel('Hemoglobin (g /dL)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.savefig(cbc_dir+"/" + "Hemoglobin Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,6])
sb.set_theme(style=None)

sb.violinplot(data=df, x='hgb', y='gender', order=['female', 'male'], orient='horizontal', palette=['pink', 'cornflowerblue']);

plt.axvline(x=12, linestyle='-', linewidth=2, color='pink', alpha=1)
plt.axvline(x=15.5, linestyle='-', linewidth=2, color='pink', alpha=1)
plt.axvline(x=13.0, linestyle='-', linewidth=2, color='blue', alpha=.25)
plt.axvline(x=17.5, linestyle='-', linewidth=2, color='blue', alpha=.25)


ticks = np.arange(5, df.hgb.max()+1, 1)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Hemoglobin Distribution By Gender', pad=10, fontsize=20)
plt.xlabel('Hemoglobin (g /dL)', labelpad=10, fontsize=15)
plt.ylabel('Gender');
plt.savefig(cbc_dir+"/" + "Hemoglobin Distribution By Gender");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,8])
sb.set_theme(style=None)

sb.regplot(data=df, x='age', y='hgb', scatter_kws={'alpha':.25}, truncate=False)

plt.title('Hemoglobin Distributio vs Age', fontsize=20, pad=15)
plt.xlabel('Age', fontsize=15, labelpad=10)
plt.ylabel('Hemoglobin (g/ dL)', fontsize=15, labelpad=10);

plt.ylim(df.hgb.quantile(.01), df.wbc.quantile(.999));
plt.savefig(cbc_dir+"/" + "Hemoglobin Distribution vs Age");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,6])
sb.set_theme(style=None)

sb.violinplot(data=df, x='hgb', y='smoker', orient='horizontal', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(5, df.hgb.max()+1, 1)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Hemoglobin Distribution vs Smoking Condition', pad=10, fontsize=20)
plt.xlabel('Hemoglobin (g /dL)', labelpad=10, fontsize=15)
plt.ylabel('Smoking Condition');
plt.savefig(cbc_dir+"/" + "Hemoglobin Distribution vs Smoking Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[25,10])
sb.set_theme(style=None)

plt.subplot(1,3,1)
sb.violinplot(data=df, y='hgb', x='chf', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(5, df.hgb.max()+1, 1)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('Hemoglobin Distribution vs CHF Condition', pad=10, fontsize=15)
plt.ylabel('Hemoglobin (g /dL)', labelpad=10, fontsize=10)
plt.xlabel('Congestive Heart Failure');

plt.subplot(1,3,2)
sb.violinplot(data=df, y='hgb', x='cad', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(5, df.hgb.max()+1, 1)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('Hemoglobin Distribution vs CAD Condition', pad=10, fontsize=15)
plt.ylabel('Hemoglobin (g /dL)', labelpad=10, fontsize=10)
plt.xlabel('Coronary artery disease');

plt.subplot(1,3,3)
sb.violinplot(data=df, y='hgb', x='mi', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(5, df.hgb.max()+1, 1)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('Hemoglobin Distribution vs MI Condition', pad=10, fontsize=15)
plt.ylabel('Hemoglobin (g /dL)', labelpad=10, fontsize=10)
plt.xlabel('Heart attack');
plt.savefig(cbc_dir+"/" + "Hemoglobin Distribution vs Heart Diseases");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,8])
sb.set_theme(style=None)

sb.regplot(data=df, x='iron', y='hgb', scatter_kws={'alpha':.25}, truncate=False)

plt.title('Hemoglobin vs iron', fontsize=20, pad=15)
plt.xlabel('Iron (mcg / dL)', fontsize=15, labelpad=10)
plt.ylabel('Hemoglobin (g/ dL)', fontsize=15, labelpad=10);

plt.xlim(df.iron.quantile(.01), df.iron.quantile(.99));
plt.savefig(cbc_dir+"/" + "Hemoglobin vs iron");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{hct}
plt.figure(figsize=[20,15])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
bins = np.arange(15, df.hct.max()+.25, .25)
plt.hist(data=df, x='hct', bins= bins, color='firebrick')

ticks = np.arange(15, df.hct.max()+2, 2)
labels = ['{:.0f}%'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Hematocrit Distribution', pad=10, fontsize=20)
plt.xlabel('Hematocrit', labelpad=10, fontsize=15)
plt.ylabel('Count');


plt.subplot(2,1,2)
sb.boxplot(data=df, x='hct', color='r')

ticks = np.arange(15, df.hct.max()+2, 2)
labels = ['{:.0f}%'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Hematocrit Distribution', pad=10, fontsize=20)
plt.xlabel('Hematocrit', labelpad=10, fontsize=15)
plt.ylabel('');
plt.savefig(cbc_dir+"/" + "Hematocrit Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[15,6])
sb.set_theme(style='whitegrid')

sb.violinplot(data=df, x='hct', y='gender', order=['female', 'male'], orient='horizontal', palette=['pink', 'cornflowerblue']);

plt.axvline(x=36, linestyle='-', linewidth=2, color='pink', alpha=1)
plt.axvline(x=48, linestyle='-', linewidth=2, color='pink', alpha=1)
plt.axvline(x=41, linestyle='-', linewidth=2, color='blue', alpha=.25)
plt.axvline(x=50, linestyle='-', linewidth=2, color='blue', alpha=.25)


ticks = np.arange(15, df.hct.max()+2, 2)
labels = ['{:.0f}%'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Hematocrit Distribution By Gender', pad=10, fontsize=20)
plt.xlabel('Hematocrit', labelpad=10, fontsize=15)
plt.ylabel('Gender');
plt.savefig(cbc_dir+"/" + "Hematocrit Distribution By Gender");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[25,10])
sb.set_theme(style='whitegrid')

plt.subplot(1,3,1)
sb.violinplot(data=df, y='hct', x='smoker', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(15, df.hct.max()+3, 3)
labels = ['{:.0f}%'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('Hematocrit Distribution vs Smoking Condition', pad=10, fontsize=15)
plt.ylabel('Hematocrit', labelpad=10, fontsize=10)
plt.xlabel('Smoking Condition');

plt.subplot(1,3,2)
sb.violinplot(data=df, y='hct', x='cancer', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(15, df.hct.max()+3, 3)
labels = ['{:.0f}%'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('Hematocrit Distribution vs Cancer', pad=10, fontsize=15)
plt.ylabel('Hematocrit', labelpad=10, fontsize=10)
plt.xlabel('Cancerous');

plt.subplot(1,3,3)
sb.violinplot(data=df, y='hct', x='copd', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(15, df.hct.max()+3, 3)
labels = ['{:.0f}%'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('Hematocrit Distribution vs COPD', pad=10, fontsize=15)
plt.ylabel('Hematocrit)', labelpad=10, fontsize=10)
plt.xlabel('Emphysema');
plt.savefig(cbc_dir+"/" + "Hematocrit Distribution vs Smoking-cancer-COPD");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[25,10])
sb.set_theme(style='whitegrid')

plt.subplot(1,3,1)
sb.violinplot(data=df, y='hct', x='chf', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(15, df.hct.max()+3, 3)
labels = ['{:.0f}%'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('Hematocrit Distribution vs CHF Condition', pad=10, fontsize=15)
plt.ylabel('Hematocrit', labelpad=10, fontsize=10)
plt.xlabel('Congestive Heart Failure');

plt.subplot(1,3,2)
sb.violinplot(data=df, y='hct', x='cad', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(15, df.hct.max()+3, 3)
labels = ['{:.0f}%'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('Hematocrit Distribution vs CAD Condition', pad=10, fontsize=15)
plt.ylabel('Hematocrit', labelpad=10, fontsize=10)
plt.xlabel('Coronary artery disease');

plt.subplot(1,3,3)
sb.violinplot(data=df, y='hct', x='mi', palette=['red', 'green', 'cornflowerblue']);

ticks = np.arange(15, df.hct.max()+3, 3)
labels = ['{:.0f}%'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('Hematocrit Distribution vs MI Condition', pad=10, fontsize=15)
plt.ylabel('Hematocrit', labelpad=10, fontsize=10)
plt.xlabel('Heart attack');
plt.savefig(cbc_dir+"/" + "Hematocrit Distribution vs Heart diseases");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')

plt.figure(figsize=[20,8])
sb.regplot(data=df, x='iron', y='hct', scatter_kws={'alpha':.25}, truncate=False)

plt.title('Hematocrit vs iron', fontsize=20, pad=15)
plt.xlabel('Iron (mcg / dL)', fontsize=15, labelpad=10)
plt.ylabel('Hematocrit', fontsize=15, labelpad=10);

plt.xlim(df.iron.quantile(.01), df.iron.quantile(.99));
plt.savefig(cbc_dir+"/" + "Hematocrit Distribution vs Heart diseases");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{platelets}
plt.figure(figsize=[25,20])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
bins = np.arange(5, df.platelets.max()+2, 2)
plt.hist(data=df, x='platelets', bins= bins, color='firebrick')

plt.axvline(x=150, linestyle='-', linewidth=2, color='yellow', alpha=1)
plt.axvline(x=450, linestyle='-', linewidth=2, color='yellow', alpha=1)

ticks = np.arange(5, df.platelets.max()+5, 25)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Platelets Distribution', pad=10, fontsize=20)
plt.xlabel('Platelets per mcL', labelpad=10, fontsize=15)
plt.ylabel('Count');
#plt.xlim(df.platelets.quantile(.01), df.platelets.quantile(.99))

plt.subplot(2,1,2)
sb.boxplot(data=df, x='platelets', color='r')

ticks = np.arange(5, df.platelets.max()+5, 25)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Platelets Distribution', pad=10, fontsize=20)
plt.xlabel('Platelets per mcL', labelpad=10, fontsize=15)
plt.ylabel('');
plt.savefig(cbc_dir+"/" + "Platelets Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,8])
sb.set_theme(style='whitegrid')

p = sb.barplot(data=df, x='drinks_day', y='platelets', palette='viridis_r');

ticks = p.get_xticks()
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('platelets vs Drinks/Day', fontsize=20, pad=15)
plt.xlabel('Drinks/Day', fontsize=15, labelpad=10)
plt.ylabel('platelets per mcL', fontsize=15, labelpad=10);
plt.savefig(cbc_dir+"/" + "platelets vs Drinks per Day");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
plt.figure(figsize=[20,8])
sb.set_theme(style='whitegrid')

sb.violinplot(data=df, x='platelets', y='cancer', orient='horizontal', palette=['red', 'green', 'cornflowerblue']);

plt.axvline(x=150, linestyle='-', linewidth=2, color='yellow', alpha=1)
plt.axvline(x=450, linestyle='-', linewidth=2, color='yellow', alpha=1)

ticks = np.arange(5, df.platelets.max()+5, 25)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=90)
plt.title('Platelets Distribution vs Cancer', pad=10, fontsize=15)
plt.xlabel('Platelets per mcL', labelpad=10, fontsize=10)
plt.ylabel('Cancerous');
plt.savefig(cbc_dir+"/" + "Platelets Distribution vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#	{wbc, hgb, hct, and plateletes relations}
CBC_features = ['wbc', 'hgb', 'hct', 'platelets', 'iron']
g = sb.PairGrid(data=df, vars=CBC_features)
g.map_offdiag(plt.scatter, alpha=.01, color='firebrick')
g.map_diag(plt.hist, color='firebrick');
plt.savefig(cbc_dir+"/" + "pair plot for wbc, hgb, hct, platelets, iron");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
CBC_features = ['wbc', 'hgb', 'hct', 'platelets', 'iron']

plt.figure(figsize= [15,10])

correlations = df[CBC_features].corr()

sb.heatmap(correlations, cmap='vlag_r', annot=True, fmt='.2f', center = 0)

plt.title('CBC Tests Correlations  (Paired)', fontsize= 22, pad=15)
plt.xlabel('CBC Tests', fontsize= 15, labelpad=20)
plt.ylabel('CBC Tests', fontsize= 15, labelpad=20);
plt.savefig(cbc_dir+"/" + "CBC Tests Correlations  (Paired)");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#				{{{Liver Function Test (LFT)}}}

#	{alt (Alanine aminotransferase)}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,20])

plt.subplot(2,1,1)
bins = np.arange(0, df.alt.max()+1, 1)

plt.hist(data=df, x='alt', bins=bins, color='turquoise')
plt.xscale('log')

plt.axvline(x=7, linestyle='-', linewidth=2, color='yellow', alpha=1)
plt.axvline(x=56, linestyle='-', linewidth=2, color='yellow', alpha=1)

ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('ALT Distribution', pad=10, fontsize=20)
plt.xlabel('ALT (IU / L)  "Log-transformed"', fontsize=12)
plt.ylabel('count');


plt.subplot(2,1,2)
sb.boxplot(data=df, x='alt', color='turquoise')
plt.xscale('log')

plt.xticks(ticks, labels, rotation=0)
plt.title('ALT Distribution', pad=10, fontsize=20)
plt.xlabel('ALT (IU / L)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.savefig(lft_dir+"/" + "ALT Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,8])
p = sb.barplot(data=df, x='drinks_day', y='alt', palette='viridis_r');

ticks = p.get_xticks()
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('ALT vs Drinks/Day', fontsize=20, pad=15)
plt.xlabel('Drinks/Day', fontsize=15, labelpad=10)
plt.ylabel('ALT (IU / L)', fontsize=15, labelpad=10);
plt.savefig(lft_dir+"/" + "ALT vs Drinks per Day");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[25,8])

sb.violinplot(data=df, x='alt', y='chf', orient='horizontal', palette=['red', 'green', 'cornflowerblue']);
plt.xscale('log')

ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('ALT Distribution vs CHF Condition', pad=10, fontsize=15)
plt.xlabel('ALT (IU / L)', labelpad=10, fontsize=10)
plt.ylabel('Congestive Heart Failure', fontsize=15, labelpad=10);
plt.savefig(lft_dir+"/" + "ALT Distribution vs CHF Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#		{ast liver enzyme}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,20])

plt.subplot(2,1,1)
bins = np.arange(0, df.ast.max()+1, 1)

plt.hist(data=df, x='ast', bins=bins, color='turquoise')
plt.xscale('log')

plt.axvline(x=5, linestyle='-', linewidth=2, color='yellow', alpha=1)
plt.axvline(x=40, linestyle='-', linewidth=2, color='yellow', alpha=1)

ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('AST Distribution', pad=10, fontsize=20)
plt.xlabel('AST (IU / L)  "Log-transformed"', fontsize=12)
plt.ylabel('count');


plt.subplot(2,1,2)
sb.boxplot(data=df, x='ast', color='turquoise')
plt.xscale('log')

plt.xticks(ticks, labels, rotation=0)
plt.title('AST Distribution', pad=10, fontsize=20)
plt.xlabel('AST (IU / L) Log-transformed', labelpad=10, fontsize=15)
plt.ylabel('');
plt.savefig(lft_dir+"/" + "AST Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,8])
p = sb.barplot(data=df, x='drinks_day', y='ast', palette='viridis_r');

ticks = p.get_xticks()
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('AST vs Drinks/Day', fontsize=20, pad=15)
plt.xlabel('Drinks/Day', fontsize=15, labelpad=10)
plt.ylabel('AST (IU / L)', fontsize=15, labelpad=10);
plt.savefig(lft_dir+"/" + "AST vs Drinks per Day");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[25,8])

sb.violinplot(data=df, x='ast', y='chf', orient='horizontal', palette=['red', 'green', 'cornflowerblue']);
plt.xscale('log')

ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('AST Distribution vs CHF Condition', pad=10, fontsize=15)
plt.xlabel('AST (IU / L)  Log-transformed', labelpad=10, fontsize=10)
plt.ylabel('Congestive Heart Failure', fontsize=15, labelpad=10);
plt.savefig(lft_dir+"/" + "ST Distribution vs CHF Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#		{alk_phos (Alkaline phosphatase)}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,20])

plt.subplot(2,1,1)
bins = np.arange(0, df.alk_phos.max()+1, 1)

plt.hist(data=df, x='alk_phos', bins=bins, color='turquoise')
plt.xscale('log')

plt.axvline(x=20, linestyle='-', linewidth=2, color='yellow', alpha=1)
plt.axvline(x=140, linestyle='-', linewidth=2, color='yellow', alpha=1)

ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Alk_phos Distribution', pad=10, fontsize=20)
plt.xlabel('Alk_phos (IU / L)  "Log-transformed"', fontsize=15)
plt.ylabel('count');


plt.subplot(2,1,2)
sb.boxplot(data=df, x='alk_phos', color='turquoise')
plt.xscale('log')

plt.xticks(ticks, labels, rotation=0)
plt.title('Alk_phos Distribution', pad=10, fontsize=20)
plt.xlabel('Alk_phos (IU / L)  "Log-transformed', labelpad=10, fontsize=15)
plt.ylabel('');
plt.savefig(lft_dir+"/" + "Alk_phos Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,10])
sb.regplot(data=df, x='age', y='alk_phos', color='turquoise', scatter_kws={'alpha':.25}, truncate=False)
plt.yscale('log')

ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
labels = ['{}'.format(v) for v in ticks]
plt.yticks(ticks, labels)

ticks = np.arange(0, 110, 10)
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Alk_phos vs Age', fontsize=20, pad=15)
plt.xlabel('Age', fontsize=15, labelpad=10)
plt.ylabel('Alk_phos (IU / L)  "Log-transformed', fontsize=15, labelpad=10);
plt.savefig(lft_dir+"/" + "Alk_phos vs Age");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[25,8])

sb.violinplot(data=df, x='alk_phos', y='chf', orient='horizontal', palette=['red', 'green', 'cornflowerblue']);
plt.xscale('log')

ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Alk_phos Distribution vs CHF Condition', pad=10, fontsize=15)
plt.xlabel('Alk_phos (IU / L) "log-transformed"', labelpad=10, fontsize=10)
plt.ylabel('Congestive Heart Failure', fontsize=15, labelpad=10);
plt.savefig(lft_dir+"/" + "Alk_phos Distribution vs CHF Condition");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[25,8])

sb.violinplot(data=df, x='alk_phos', y='cancer', orient='horizontal', palette=['red', 'green', 'cornflowerblue']);
plt.xscale('log')

ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Alk_phos Distribution vs Cancer', pad=10, fontsize=15)
plt.xlabel('Alk_phos (IU / L) "log-transformed"', labelpad=10, fontsize=10)
plt.ylabel('Cancer', fontsize=15, labelpad=10);
plt.savefig(lft_dir+"/" + "Alk_phos Distribution vs Cancer");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
LFT_features = ['alt', 'ast', 'alk_phos']
g = sb.PairGrid(data=df, vars=LFT_features)
g.map_offdiag(plt.scatter, alpha=.01)
g.map_diag(plt.hist);
plt.savefig(lft_dir+"/" + "pair plot for alt-ast-alk_phos");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
LFT_features = ['alt', 'ast', 'alk_phos']

plt.figure(figsize= [14,7])

correlations = df[LFT_features].corr()

sb.heatmap(correlations, cmap='vlag_r', annot=True, fmt='.2f', center = 0)

plt.title('LFT Tests Correlations  (Paired)', fontsize= 22, pad=15)
plt.xlabel('LFT Tests', fontsize= 15, labelpad=20)
plt.ylabel('LFT Tests', fontsize= 15, labelpad=20);
plt.savefig(lft_dir+"/" + "LFT Tests Correlations  (Paired)");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#				{{{{{{{{{{{{{{{{{{Kidney Function Test (KFT)}}}}}}}}}}}}}}}}}}
#		{bun Blood urea Nitrogen (BUN)}
sb.set_theme(style='whitegrid')
plt.figure(figsize=[25,20])

plt.subplot(2,1,1)
bins = np.arange(1, df.bun.max()+1, 1)
plt.hist(data=df, x='bun', bins= bins, color='m')
plt.xscale('log')

plt.axvline(x=7, linestyle='-', linewidth=2, color='yellow', alpha=1)
plt.axvline(x=21, linestyle='-', linewidth=2, color='yellow', alpha=1)

ticks = [1,2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70,100]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels, rotation=0)

plt.title('BUN Distribution', pad=10, fontsize=20)
plt.xlabel('BUN  (mg/dL) "log-transfered"', labelpad=10, fontsize=15)
plt.ylabel('Count');


plt.subplot(2,1,2)
sb.boxplot(data=df, x='bun', color='m')
plt.xscale('log')

plt.xticks(ticks, labels, rotation=0)
plt.title('BUN Distribution', pad=10, fontsize=20)
plt.xlabel('BUN  (mg/dL) "log-transformed"', labelpad=10, fontsize=15)
plt.ylabel('');
plt.savefig(kft_dir+"/" + "BUN Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,10])
sb.regplot(data=df, x='age', y='bun', color='m', scatter_kws={'alpha':.25}, truncate=False)
plt.yscale('log')

ticks = [1,2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70,100]
labels = ['{}'.format(v) for v in ticks]
plt.yticks(ticks, labels)

ticks = np.arange(0, 110, 10)
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Blood urea Nitrogen (BUN) vs Age', fontsize=20, pad=15)
plt.xlabel('Age', fontsize=15, labelpad=10)
plt.ylabel('BUN  (mg/dL) "log-transformed"', fontsize=15, labelpad=10);
plt.savefig(kft_dir+"/" + "Blood urea Nitrogen (BUN) vs Age");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[25,10])

plt.subplot(1,3,1)
sb.violinplot(data=df, y='bun', x='chf', palette=['red', 'green', 'cornflowerblue']);
plt.yscale('log')

ticks = [1,2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70,100]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.yticks(ticks, labels)
plt.title('BUN Distribution vs CHF Condition', pad=10, fontsize=15)
plt.ylabel('BUN  (mg/dL) "log-transformed"', labelpad=10, fontsize=10)
plt.xlabel('Congestive Heart Failure');
plt.ylim(1)


plt.subplot(1,3,2)
sb.violinplot(data=df, y='bun', x='hypertension', palette=['red', 'green', 'yellow', 'cornflowerblue']);
plt.yscale('log')

plt.yticks(ticks, labels)
plt.title('BUN Distribution vs Hypertension', pad=10, fontsize=15)
plt.ylabel('BUN  (mg/dL) "log-transformed"', labelpad=10, fontsize=10)
plt.xlabel('Hypertension');
plt.ylim(1)


plt.subplot(1,3,3)
sb.violinplot(data=df, y='bun', x='diabetes', palette=['red', 'lightsalmon', 'green', 'cornflowerblue']);
plt.yscale('log')

plt.yticks(ticks, labels)
plt.title('BUN Distribution vs Diabetes', pad=10, fontsize=15)
plt.ylabel('BUN  (mg/dL) "log-transformed"', labelpad=10, fontsize=10)
plt.xlabel('Diabetic')
plt.ylim(1);
plt.savefig(kft_dir+"/" + "BUN Distribution vs CHF-Hypertension-Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#		{cr (Serum Creatinine test (Cr))}
plt.figure(figsize=[20,18])
sb.set_theme(style='whitegrid')

plt.subplot(2,1,1)
bins = np.arange(.1, df.cr.max()+.05, .05)
plt.hist(data=df, x='cr', bins= bins, color='m')
plt.xscale('log')
plt.axvline(x= .5, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 1.2, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = [0.1, 0.2, 0.3, 0.4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)
plt.title('Serum Creatinine Distribution', pad=10, fontsize=20)
plt.xlabel('Serum Creatinine (mg / dL) "log-transformed"', labelpad=10, fontsize=15)
plt.ylabel('Count');


plt.subplot(2,1,2)

sb.boxplot(data=df, x='cr', color='m')
plt.xscale('log')

plt.xticks(ticks, labels)
plt.title('Serum Creatinine Distribution', pad=10, fontsize=20)
plt.xlabel('Serum Creatinine (mg / dL) "log-transformed"', labelpad=10, fontsize=15)
plt.ylabel('');
plt.savefig(kft_dir+"/" + "Serum Creatinine Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,6])

sb.violinplot(data=df, x='cr', y='gender', order=['female', 'male'], orient='horizontal', palette=['pink', 'cornflowerblue']);
plt.xscale('log')

plt.axvline(x= .5, linestyle='-', linewidth=2, color='pink', alpha=.99)
plt.axvline(x= 1, linestyle='-', linewidth=2, color='pink', alpha=.99)
plt.axvline(x= .7, linestyle='-', linewidth=2, color='cornflowerblue', alpha=.99)
plt.axvline(x= 1.2, linestyle='-', linewidth=2, color='cornflowerblue', alpha=.99)

ticks = [0.1, 0.2, 0.3, 0.4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = ['{}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Serum Creatinine Distribution by Gender', pad=10, fontsize=15)
plt.xlabel('Serum Creatinine  (mg/dL) "log-transformed"', labelpad=10, fontsize=10)
plt.ylabel('Gender');
plt.xlim(.1);
plt.savefig(kft_dir+"/" + "Serum Creatinine Distribution by Gender");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,10])

plt.subplot(1,2,1)
sb.violinplot(data=df, y='cr', x='hypertension', palette=['red', 'green', 'yellow', 'cornflowerblue']);
plt.yscale('log')

ticks = [0.1, 0.2, 0.3, 0.4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = ['{}'.format(v) for v in ticks]

plt.yticks(ticks, labels)
plt.title('Serum Creatinine Distribution vs Hypertension', pad=10, fontsize=15)
plt.ylabel('Serum Creatinine  (mg/dL) "log-transformed"', labelpad=10, fontsize=10)
plt.xlabel('Hypertension');
plt.ylim(.1)


plt.subplot(1,2,2)
sb.violinplot(data=df, y='cr', x='diabetes', palette=['red', 'lightsalmon', 'green', 'cornflowerblue']);
plt.yscale('log')

plt.yticks(ticks, labels)
plt.title('Serum Creatinine Distribution vs Diabetes', pad=10, fontsize=15)
plt.ylabel('Serum Creatinine  (mg/dL) "log-transformed"', labelpad=10, fontsize=10)
plt.xlabel('Diabetic')
plt.ylim(.1);
plt.savefig(kft_dir+"/" + "Serum Creatinine Distribution vs Hypertension-Diabetes");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
KFT_features = ['age', 'bun', 'cr']
g = sb.PairGrid(data=df, vars=KFT_features)
g.map_offdiag(plt.scatter, alpha=.01, color='m')
g.map_diag(plt.hist, color='m');
plt.savefig(kft_dir+"/" + "pair plot age-BUN-Cr");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
KFT_features = ['age', 'bun', 'cr']

plt.figure(figsize= [14,7])

correlations = df[KFT_features].corr()

sb.heatmap(correlations, cmap='coolwarm_r', annot=True, fmt='.2f', center = 0)

plt.title('KFT Tests  + Age Correlations  (Paired)', fontsize= 22, pad=15)
plt.xlabel('KFT Tests + Age', fontsize= 15, labelpad=20)
plt.ylabel('KFT Tests + Age', fontsize= 15, labelpad=20);
plt.savefig(kft_dir+"/" + "KFT Tests Correlations with age (Paired)");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#					{{{{{{{{{{{{{{Comprehensive metabolic panel (CMP)}}}}}}}}}}}}}}

#	sodium
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(90, df.sodium.max()+1, 1)
plt.hist(data=df, x='sodium', bins= bins, color=sb.color_palette('YlOrBr', 30)[10])

plt.axvline(x= 135, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 145, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(90, df.sodium.max()+5, 5)
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Sodium in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Sodium in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df.sodium.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='sodium', color=sb.color_palette('YlOrBr', 30)[10])

plt.xticks(ticks, labels)
plt.title('Sodium in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Sodium in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df.sodium.min(),);
plt.savefig(cmp_dir+"/" + "Sodium in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(1.5, df.potassium.max()+.1, .1)
plt.hist(data=df, x='potassium', bins= bins, color=sb.color_palette('YlOrBr', 30)[12])

plt.axvline(x= 3.6, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 5.2, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(1.5, df.potassium.max()+.2, .2)
labels = ['{:.1f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Potassium in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Potassium in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df.potassium.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='potassium', color=sb.color_palette('YlOrBr', 30)[12])

plt.xticks(ticks, labels)
plt.title('Potassium in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('potassium in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df.potassium.min(),);
plt.savefig(cmp_dir+"/" + "Potassium in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(10, df.bicarb.max()+1, 1)
plt.hist(data=df, x='bicarb', bins= bins, color=sb.color_palette('YlOrBr', 30)[14])

plt.axvline(x= 23, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 30, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(10, df.bicarb.max()+2, 2)
labels = ['{:.1f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Bicarbonate in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Bicarbonate in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df.bicarb.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='bicarb', color=sb.color_palette('YlOrBr', 30)[14])

plt.xticks(ticks, labels)
plt.title('Bicarbonate in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Bicarbonate in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df.bicarb.min(),);
plt.savefig(cmp_dir+"/" + "Bicarbonate in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(5, df.ca.max()+.1, .1)
plt.hist(data=df, x='ca', bins= bins, color=sb.color_palette('YlOrBr', 30)[16])

plt.axvline(x= 8.6, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 10.3, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(5, df.ca.max()+.25, .25)
labels = ['{:.2f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Calcium in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Calcium in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df.ca.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='ca', color=sb.color_palette('YlOrBr', 30)[16])

plt.xticks(ticks, labels)
plt.title('Calcium in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Calcium in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df.ca.min(),);
plt.savefig(cmp_dir+"/" + "Calcium in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(1, df.phos.max()+.1, .1)
plt.hist(data=df, x='phos', bins= bins, color=sb.color_palette('YlOrBr', 30)[18])

plt.axvline(x= 2.8, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 7, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(1, df.phos.max()+.25, .25)
labels = ['{:.2f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Phosphorus in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Phosphorus in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df.phos.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='phos', color=sb.color_palette('YlOrBr', 30)[18])

plt.xticks(ticks, labels)
plt.title('Phosphorus in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Phosphorus in Blood (mEq / L)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df.phos.min(),);
plt.savefig(cmp_dir+"/" + "Phosphorus in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(0, df.t_bilirubin.max()+.1, .1)
plt.hist(data=df, x='t_bilirubin', bins= bins, color=sb.color_palette('YlOrBr', 30)[20])

plt.axvline(x= 1, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 1.2, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(0, df.t_bilirubin.max()+.25, .25)
labels = ['{:.2f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Total Bilirubin in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Total Bilirubin in Blood (mg / dL)', labelpad=10, fontsize=15)
plt.ylabel('Count');
#plt.xlim(df.t_bilirubin.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='t_bilirubin', color=sb.color_palette('YlOrBr', 30)[20])

plt.xticks(ticks, labels)
plt.title('Total Bilirubin in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Total Bilirubin in Blood (mg / dL)', labelpad=10, fontsize=15)
plt.ylabel('');
#plt.xlim(df.t_bilirubin.min(),);
plt.savefig(cmp_dir+"/" + "Total Bilirubin in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(0, df.alb.max()+.1, .1)
plt.hist(data=df, x='alb', bins= bins, color=sb.color_palette('YlOrBr', 30)[22])

plt.axvline(x= 3.4, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 5.4, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(1, df.alb.max()+.25, .25)
labels = ['{:.2f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Albumin in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Albumin in Blood (g / dL)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df.alb.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='alb', color=sb.color_palette('YlOrBr', 30)[22])

plt.xticks(ticks, labels)
plt.title('Albumin in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Albumin in Blood (g / dL)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df.alb.min(),);
plt.savefig(cmp_dir+"/" + "Albumin in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(1, df.t_protein.max()+.1, .1)
plt.hist(data=df, x='t_protein', bins= bins, color=sb.color_palette('YlOrBr', 30)[24])

plt.axvline(x= 6, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 8, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(1, df.t_protein.max()+.5, .5)
labels = ['{:.1f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Total protein in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Total protein in Blood (g / dL)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df.t_protein.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='t_protein', color=sb.color_palette('YlOrBr', 30)[22])

plt.xticks(ticks, labels)
plt.title('Total protein in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Total protein in Blood (g / dL)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df.t_protein.min(),);
plt.savefig(cmp_dir+"/" + "Albumin in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(.1, df.glob.max()+.1, .1)
plt.hist(data=df, x='glob', bins= bins, color=sb.color_palette('YlOrBr', 30)[26])

plt.axvline(x= 2, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 3.5, linestyle='-', linewidth=2, color='yellow', alpha=.99)

ticks = np.arange(0, df.glob.max()+.25, .25)
labels = ['{:.2f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Globulins in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Globulins in Blood (g / dL)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df.glob.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='glob', color=sb.color_palette('YlOrBr', 30)[26])

plt.xticks(ticks, labels)
plt.title('Globulins in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Globulins in Blood (g / dL)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df.glob.min(),);
plt.savefig(cmp_dir+"/" + "Globulins in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(1, df.glucose.max()+5, 5)
plt.hist(data=df, x='glucose', bins= bins, color=sb.color_palette('YlOrBr', 30)[28])
plt.xscale('log')

plt.axvline(x= 70, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 140, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 200, linestyle='-', linewidth=2, color='red', alpha=.99)
plt.axvline(x= 600, linestyle='-', linewidth=2, color='darkred', alpha=.99)

ticks = [1,2,5,10,20,50,60,70,80,90,100,120,140,160,180,200,500,1000]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Glucose in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Glucose in Blood (mg / dL)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df.glucose.min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='glucose', color=sb.color_palette('YlOrBr', 30)[28])
plt.xscale('log')

plt.xticks(ticks, labels)
plt.title('Glucose in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Glucose in Blood (mg / dL)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df.glucose.min(),);
plt.savefig(cmp_dir+"/" + "Glucose in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[20,18])

plt.subplot(2,1,1)
bins = np.arange(1, df.glucose.max()+5, 5)
plt.hist(data=df, x='glucose.1', bins= bins, color=sb.color_palette('YlOrBr', 31)[30])
plt.xscale('log')

plt.axvline(x= 70, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 100, linestyle='-', linewidth=2, color='yellow', alpha=.99)
plt.axvline(x= 125, linestyle='-', linewidth=2, color='red', alpha=.99)

ticks = [1,2,5,10,20,50,60,70,80,90,100,120,140,160,180,200,500]
labels = ['{:.0f}'.format(v) for v in ticks]
plt.xticks(ticks, labels)

plt.title('Fasting Glucose in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Fasting Glucose in Blood (mg / dL)', labelpad=10, fontsize=15)
plt.ylabel('Count');
plt.xlim(df['glucose.1'].min(),);

plt.subplot(2,1,2)

sb.boxplot(data=df, x='glucose.1', color=sb.color_palette('YlOrBr', 31)[30])
plt.xscale('log')

plt.xticks(ticks, labels)
plt.title('Fasting Glucose in Blood Distribution', pad=10, fontsize=20)
plt.xlabel('Fasting Glucose in Blood (mg / dL)', labelpad=10, fontsize=15)
plt.ylabel('');
plt.xlim(df['glucose.1'].min(),);
plt.savefig(cmp_dir+"/" + "Fasting Glucose in Blood Distribution");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
plt.figure(figsize=[25,15])

plt.subplot(1,2,1)
sb.violinplot(data=df, y='glucose', x='diabetes', palette=['red', 'lightsalmon', 'green', 'cornflowerblue']);
plt.yscale('log')

ticks = [1,2,5,10,20,50,60,70,80,90,100,120,140,160,180,200, 300, 400,500,1000]
labels = ['{}'.format(v) for v in ticks]

plt.yticks(ticks, labels)
plt.title('Glucose Distribution vs Diabetes Disease', pad=10, fontsize=15)
plt.ylabel('Glucose  (mg/dL) "log-transformed"', labelpad=10, fontsize=10)
plt.xlabel('Diabetic');
plt.ylim(df['glucose'].min(),df['glucose'].max());


plt.subplot(1,2,2)
sb.violinplot(data=df, y='glucose.1', x='diabetes', palette=['red', 'lightsalmon', 'green', 'cornflowerblue']);
plt.yscale('log')

ticks = [1,2,5,10,20,50,60,70,80,90,100,120,140,160,180,200,300,400,500]
labels = ['{}'.format(v) for v in ticks]

plt.yticks(ticks, labels)
plt.title('Fasting Glucose Distribution vs Diabetes Disease', pad=10, fontsize=15)
plt.ylabel('Fasting Glucose  (mg/dL) "log-transformed"', labelpad=10, fontsize=10)
plt.xlabel('Diabetic')
plt.ylim(df['glucose.1'].min(),df['glucose.1'].max());
plt.savefig(cmp_dir+"/" + "Glucose & Fasting Glucose Distribution vs Diabetes Disease");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
sb.set_theme(style='whitegrid')
CMP_features = ['potassium', 'chloride', 'bicarb', 'ca', 'phos', 't_bilirubin', 'alb', 
                't_protein', 'glob', 'glucose', 'glucose.1',]

plt.figure(figsize= [30,25])

correlations = df[CMP_features].corr()

sb.heatmap(correlations, cmap='coolwarm_r', annot=True, fmt='.2f', center = 0)

plt.title('CMP Tests Correlations  (Paired)', fontsize= 22, pad=15)
plt.xlabel('CMP Tests', fontsize= 15, labelpad=20)
plt.ylabel('CMP Tests', fontsize= 15, labelpad=20);
plt.savefig(cmp_dir+"/" + "CMP Tests Correlations  (Paired)");
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Finished