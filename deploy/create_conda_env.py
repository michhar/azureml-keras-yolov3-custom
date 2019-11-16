from azureml.core.runconfig import CondaDependencies

cd = CondaDependencies.create()
cd.add_pip_package('tensorflow==1.13.1')
cd.add_pip_package('keras==2.2.4')
cd.add_pip_package('matplotlib==3.1.1')
cd.add_pip_package('opencv-python==4.1.1.26')
cd.add_pip_package('Pillow')

cd.save_to_file(base_directory='./', conda_file_path='keras_env.yml')

print(cd.serialize_to_string())
