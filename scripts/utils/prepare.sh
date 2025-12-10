mkdir -p checkpoints/t2m
cd checkpoints/t2m

echo -e "Downloading pretrained MoMask"
gdown --fuzzy https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing

echo -e "Unzipping humanml3d_models.zip"
unzip humanml3d_models.zip

echo -e "Cleaning humanml3d_models.zip"
rm humanml3d_models.zip

echo -e "Creating symbolic links"

ln -sr rvq_nq6_dc512_nc512_noshare_qdp0.2 rvq
ln -sr tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw rtrans
ln -sr t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns mtrans

ln -sr rvq/model/net_best_fid.tar rvq/model/base.tar
ln -sr mtrans/model/latest.tar mtrans/model/base.tar
ln -sr rtrans/model/net_best_fid.tar rtrans/model/base.tar

echo -e "Downloading pretrained BAMM"
gdown --fuzzy https://drive.google.com/file/d/1vo0PcYOHzCdPoDk5SpBA1llvK_50GTlv/view?usp=sharing
echo -e "Unzipping 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd.zip"
unzip 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd.zip

echo -e "Cleaning 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd.zip"
rm 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd.zip
mv 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd bamm
ln -sr bamm/model/latest.tar bamm/model/base.tar

mv bamm/opt.txt bamm/opt.txt.bak
cat bamm/opt.txt.bak | sed 's/rvq_nq6_dc512_nc512_noshare_qdp0.2/rvq/' | sed 's/2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd/bamm/' | sed 's|log/t2m|checkpoints/t2m|' > bamm/opt.txt

cd ../../
ln -sr checkpoints/t2m/ checkpoints/HumanML3D

mkdir -p dataset/