wget https://www.dropbox.com/sh/pdzugtmbgyjdnpg/AAAg4eRgLkh9aJlvFvp_S4Lra?dl=1 -O dropbox_data.zip
unzip dropbox_data.zip -d dropbox_data
rm dropbox_data.zip
cp -r dropbox_data/* data/
rm -r dropbox_data