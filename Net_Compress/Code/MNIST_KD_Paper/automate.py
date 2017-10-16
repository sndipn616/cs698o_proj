import os
import sys
import mnist_paper_teacher as T
import mnist_paper_student as S

def main():
	T.pre_train_teacher()
	S.pre_train_student()

if __name__ == '__main__':
	main()