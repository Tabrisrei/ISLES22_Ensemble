mv /etc/apt/sources.list /etc/apt/sources.list.bak
# echo "deb http://mirrors.163.com/debian/ jessie main non-free contrib" >> /etc/apt/sources.list
# echo "deb http://mirrors.163.com/debian/ jessie-proposed-updates main non-free contrib" >>/etc/apt/sources.list
# echo "deb-src http://mirrors.163.com/debian/ jessie main non-free contrib" >>/etc/apt/sources.list
# echo "deb-src http://mirrors.163.com/debian/ jessie-proposed-updates main non-free contrib" >>/etc/apt/sources.list
# echo "" > /etc/apt/sources.list
echo "deb http://mirrors.aliyun.com/debian buster main" >> /etc/apt/sources.list ;
echo "deb http://mirrors.aliyun.com/debian-security buster/updates main" >> /etc/apt/sources.list ;
echo "deb http://mirrors.aliyun.com/debian buster-updates main" >> /etc/apt/sources.list ;
# apt-get update | 'bin/echo' -e "\ny\n"
# apt-get install vim | 'bin/echo' -e "\ny\n"

echo y | apt-get update
echo y | apt-get install vim