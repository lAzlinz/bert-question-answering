from transformers import pipeline
from train import output_dir as model_dir

model_name: str = 'my_model'
model_path: str = model_dir + model_name + '/checkpoint-4'

question = "Where to get the BUCET Form online?" # "How can I reschedule my appointment date after I passed the BUCET exam?"
contexta = "The appointment date can be adjusted by going to the website https://buao.bicol-u.edu.ph/bucet_online/retrieve.php."

contexts = [
    "The specific requirements are as follows. Correctly and completely accomplished BUCET Application Form. Two (2) copies of 2”x2” recent, unedited or unfiltered studio- taken photograph of the applicant in white background with applicant’s signature and name tag. Grade 9 to Grade 11 High School academic ratings affixed at the reverse side of the BUCET Application Form and duly signed by the School Principal and has a school dry seal. Authenticated Photocopies of Grade 9-11 SF10 (Formerly Form 137) and Grade 12 SF9 (Formerly Form 138) FOR SENIOR HIGH SCHOOL GRADUATES who have not taken any college subject. For ALS graduates, submit your ALS Certificate of Rating and Certificate of Completion. DO NOT SUBMIT the original copies unless required by the office to be presented for authentication purposes. Photocopy of Person with Disability (PWD) ID pursuannt to RA No. 7277(if applicable). Photocopy of parent's Solo Parent ID pursuant to RA No. 11861 (if applicable). Copy of Parent’s latest Income Tax Return (ITR) or Certificate of Tax Exemption from the Bureau of Internal Revenue (BIR). Accomplished NCIP - COC Form 4 or downloadable Certificate of Membership from the National Commission on Indigenous Peoples (NCIP) Regional Office, if applicant is a member of ethno-linguistic groups (Indigenous Peoples groups). Certification from the Barangay if applicant is a resident of or comes from Geographically-Isolated and Disadvantaged Areas, (GIDA) pursuant to RA No. 11148, please refer to this link GIDA List 2022. Certification from DSWD if member of 4P’s (Pantawid Pamilyang Pilipino Program). One long-sized window mailing envelope with Php 50.00 worth of mailing stamps from Postal Office attached on the upper right-hand corner for mailing of results. The Online BUCET Application Form can be accessed through the BUAO website: buao.bicol-u.edu.ph. Make sure to have a stable internet connection. Whatever input that will be submitted by the applicant is final and cannot be changed. If any problem occurs in filling up the forms, please email: buao.bucet@gmail.com;"
    ,"Incoming freshmen are required to take the BUCET. It is a three-hour examination consisting of sub-tests in Language Proficiency in English, Mathematics, Science and Reading Comprehension. Admission shall be based on the applicant's Bicol University College Entrance Test (BUCET) Composite Rating consisting of the BUCET score and his/her general weighted average (GWA) in grades 9 to 11. The Online BUCET Application Form can be accessed through the BUAO website: buao.bicol-u.edu.ph. Make sure to have a stable internet connection. Whatever input that will be submitted by the applicant is final and cannot be changed. If any problem occurs in filling up the forms, please email: buao.bucet@gmail.com;"
]

question_answerer = pipeline("question-answering", model=model_path)

while True:
    q: str = input("Question: ")
    if q == 'q':
        break

    for context in contexts:
        ans = question_answerer(question=q, context=context)
        if(ans['score'] >= 0.1):
            print(ans['answer'])