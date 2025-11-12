from vidstab import VidStab

stabilizer = VidStab()
stabilizer.stabilize(input_path='in/irl/iphone/y.mov', 
                     output_path='out/y_stabilized.avi', 
                     border_size=-10)
