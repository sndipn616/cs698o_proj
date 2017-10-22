import os
import sys
import mnist_teacher as T
import mnist_student as S

teacher_model = 'mnist_teacher'
student_models = ['1_mnist_student_init', '2_mnist_student', '3_mnist_student_KD', '4_mnist_student_KD_adv']
'''
1_mnist_student_init - initial student model with which other student models are initialized
2_mnist_student - training student initialized with above model without KD
3_mnist_student_KD - training student initialized with above model with KD
4_mnist_student_KD_adv - training student initialized with above model with adversarial KD
'''

def main():
	T.test_teacher(teacher_model)
	for student_model in student_models:
		S.test_student(student_model)

if __name__ == '__main__':
	main()