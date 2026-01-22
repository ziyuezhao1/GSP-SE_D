"""
Experiment for spectral method.
"""
import pathlib
import numpy as np
import experiment as ep
def experiment_gaussian(dirpath):
    # Parameters
    TEST_NAME = "experiment_gaussian"
    ns = [500]
    sigmas1=[1]#np.arange(0.0,26, 0.5)
    sigmas2=np.arange(0.0,8, 0.2)
    rep_no = 25

    # Messaging parameter
    trial_no = len(ns) * len(sigmas1) *len(sigmas2) * rep_no;


    # Set up CSV file
    fields = ["n", "s1","s2","q", "rep_no","eigen_mat_rerr","eigen_mat_terr","eigen_mat_gerr","eigen_mat_ev_time","eigen_mat_cal_translation_time"]
    f, dw = ep.openCSVFile(dirpath, TEST_NAME, fields = fields)

    # Experimental loop
    row = {}
    count = 0
    for n in ns:
        row['n'] = n
        for s1 in sigmas1:
            row['s1'] = s1
            for s2 in sigmas2:
                row["s2"]=s2
                for rep in range(rep_no):
                    row["rep_no"]=rep                    
                            
        
                    # Run experiment
                    rerr_eigen_mat, terr_eigen_mat,mat_eigen_times,gerr_eigen_mat= ep.experiment(n=n,sigma1=s1,sigma2=s2)
                    row["eigen_mat_ev_time"] = mat_eigen_times["eigen_ev_time"]
                    row["eigen_mat_cal_translation_time"] = mat_eigen_times["eigen_cal_translation_time"]                        
                            
                    # Process data
                    row["eigen_mat_rerr"] = rerr_eigen_mat
                    row["eigen_mat_terr"] = terr_eigen_mat
                    row["eigen_mat_gerr"] = gerr_eigen_mat
        
                            
                    # Increase the coutner
                    count+=1
                    dw.writerow(row)
        
                    ep.logmsg("Trial {:d} of {:d} completed for sigma1 is {:f} sigma2 is {:f}.\n\t\tmean rerr\tmean terr\tmean gerr\t ev_time\tcal_tran_time\n\t ASE\t{:.15}\t\t{:.15}\t\t{:.8}\t\t{:.4}".format(count, trial_no,s1,s2,row["eigen_mat_rerr"], row["eigen_mat_terr"],row["eigen_mat_gerr"],row["eigen_mat_ev_time"],row["eigen_mat_cal_translation_time"]))
                                

    f.close()
if __name__=="__main__":
    dirpath = pathlib.Path(__file__).parent.resolve()
    path=str(dirpath)+"\compared_experiment_results_with_trans"
    print("Writing directory: \t", path)
    experiment_gaussian(dirpath=path)
