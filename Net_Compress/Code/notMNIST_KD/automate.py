import os
import sys
import notmnist_teacher as T
import notmnist_student as S
import notmnist_KD2 as S_KD

def main():
	# T.pre_train_teacher()
	# S.pre_train_student()
	S_KD.train_student_KD()

if __name__ == '__main__':
	main()