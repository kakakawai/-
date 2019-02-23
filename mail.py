# -*- coding:utf-8 -*-
#Author:Mirror
#Create Date: 2018.7.15
#Modify Date: 2018.7.16

import smtplib
from email.mime.text import MIMEText
from email.header import Header


def Send_Email(subject,message,receivers):
    mailHost = "smtp.qq.com"
    mailUser = "609543803@qq.com"
    mailPass = "xeftwicqwfcnbdci"

    sender = mailUser
    #receivers = ["609543803@qq.com"]

    message = MIMEText(message,"plain","utf-8")
    message["From"] = Header(mailUser,"utf-8")
    #message["To"] = Header(receivers,'utf-8')
    #subject = "SMTP TEST"
    message["Subject"] = Header(subject,'utf-8')

    smtpObj = smtplib.SMTP()
    smtpObj.connect(mailHost,587)
    smtpObj.starttls()
    smtpObj.login(mailUser,mailPass)
    smtpObj.sendmail(sender,receivers,message.as_string())
    print("Send!")

if __name__ == "__main__":
    subject = "[BTC-HMM]2018-7-16-16:[8]"
    message = "[aa]Next 4 hours forecast encode: 8"
    Send_Email(subject, message, ["609543803@qq.com"])
