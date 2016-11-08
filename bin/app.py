import web

urls = (
    '/home', 'Home',
    '/test', 'Test'
)

app = web.application(urls, globals())

render = web.template.render('templates/', base='layout')

class Home(object):
    def GET(self):
        return render.home()

class Test(object):
    def GET(self):
        return render.test()

if __name__ == "__main__":
    app.run()
