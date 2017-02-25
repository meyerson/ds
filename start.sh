#!/bin/bash
if [[ $1 == "shell" ]]
then 
    exec /bin/bash
else
    supervisord -n -c /opt/app/conf/supervisord.conf 
fi