import os
import sys
import mnist_teacher as T
import mnist_student_fitnet as S
import mnist_fitnet_guided as S1
import mnist_fitnet_guided_KD as S2

def main():
	T.pre_train_teacher()
	S.pre_train_student()
	S1.train_student_KD()
	S2.train_student_KD(flag=1)
	S2.train_student_KD(flag=2)

if __name__ == '__main__':
	main()