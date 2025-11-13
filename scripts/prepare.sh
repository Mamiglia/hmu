mkdir -p checkpoints/t2m
cd checkpoints/t2m

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