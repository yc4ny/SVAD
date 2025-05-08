echo "In order to run this tool, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

wget https://huggingface.co/xg-chu/GAGAvatar_track/resolve/main/track_resources.tar ./track_resources.tar
tar -xvf track_resources.tar -C assets/
rm -r track_resources.tar
