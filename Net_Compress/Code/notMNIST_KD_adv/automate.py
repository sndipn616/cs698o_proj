import os
import sys
import notmnist_teacher as T
import notmnist_student as S
import discriminator_create_data as DC
import discriminator_train as DT
import notmnist_KD2 as S_KD

def main():
	# T.pre_train_teacher()
	# S.pre_train_student()
	# DC.create_disc_data()
	# DT.train_discriminator()
	S_KD.train_student_KD_adv()

if __name__ == '__main__':
	main()