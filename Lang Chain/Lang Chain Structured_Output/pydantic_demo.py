from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = "nikit_bisht"      #default values
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt=0, lt=10, default=5, description='a value repressenting the cgpa')


#data validation exist
new_student = {'age':'45'}

student = Student(**new_student)

print(student)
student_dict = dict(student)
student_json = student.model_dump_json()
# print(type(student))