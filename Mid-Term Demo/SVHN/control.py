import os
import sys
import svhn_teacher as T
import svhn_student as S

teacher_model = 'svhn_teacher'
student_models = ['1_svhn_student_init', '2_svhn_student', '3_svhn_student_KD', '4_svhn_student_KD_adv']
'''
1_svhn_student_init - initial student model with which other student models are initialized
2_svhn_student - training student initialized with above model without KD
3_svhn_student_KD - training student initialized with above model with KD
4_svhn_student_KD_adv - training student initialized with above model with adversarial KD
'''

def main():
	T.test_teacher(teacher_model)
	for student_model in student_models:
		S.test_student(student_model)

if __name__ == '__main__':
	main()