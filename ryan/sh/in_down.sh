#!/bin/bash

cd ..

#cai gpuls
#rsync -avzP companyai8way@10.64.50.108:/home/companyai8way/ryan/dssm/data_out/ ./data_out/
rsync -avzP companyai8way@10.64.50.108:/home/companyai8way/david/bert/valid ./data_in/
# rsync -avz companyai8way@10.64.50.108:/home/companyai8way/ryan/dssm_tf/lib/ ./lib/test/

# rsync -avz companyai8way@10.64.41.95:/home/companyai8way/ryan/prediction_pt/checkpoint/ ./checkpoint/

#m202서버
# rsync -avz irteam@10.116.92.201::R/home1/irteam/dssm/ ./data/

#rsync -avz irteam@10.108.15.73::R/home1/irteam/ryan/dssm/* ./

# rsync -avz irteam@10.108.15.73::R/home1/irteam/ryan/dssm/checkpoint/ ./checkpoint/
# rsync -avz irteam@10.108.15.73::R/home1/irteam/ryan/dssm/result/ ./result/
# rsync -avz irteam@10.108.15.73::R/home1/irteam/ryan/dssm/blog_base_data.pkl ./result/
# rsync -avz irteam@10.108.15.73::R/home1/irteam/ryan/dssm/data/kin/q_q_tapi_kin.pkl ./data/kin/
# rsync -avz irteam@10.108.15.73::R/home1/irteam/ryan/dssm/dssm_result.xlsx ./
# rsync -avz irteam@10.108.15.73::R/home1/irteam/ryan/dssm/dssm_result.xlsx ./
# rsync -avz irteam@10.108.15.73::R/home1/irteam/ryan/dssm/dssm_total_result.xlsx ./
# rsync -avz irteam@10.108.15.73::R/home1/irteam/ryan/dssm/data/cafe/ ./data/cafe/
