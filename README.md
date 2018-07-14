# edward

## usage

```
# run build command
docker build -t edward:latest -f docker/Dockerfile .
docker run -itd -p 8080:8080 -v [this project root path in your host machine]:/home/app-user/work edward:latest
```

Access http://localhost:8080 on your browser.

Input 'token' to password form , then you can use jupyter notebook.