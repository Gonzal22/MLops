# MLops
Tareas de Mlops
#Instrucciones para correr 

# Corrida 1: Defecto
python train.py

# Corrida 2: C=0.1
python train.py --c_param 0.1 --solver lbfgs

# Corrida 3: C=10 , solver = liblinear
python train.py --c_param 10.0 --solver liblinear

# Corrida 4: C= 0.5 , solver = newton-cg
python train.py --c_param 0.5 --solver newton-cg

# Corrida 5: semilla = 99
python train.py --c_param 1.0 --solver lbfgs --random_state 99

#Corrida 6 = Defecto   Para probar que obtuve el mismo resultado
Python train.py
