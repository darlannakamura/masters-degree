# Hello!

This is the default report template! 

if you want to generate a different report, you need to call your report like this:

```py
report = Report(template='PATH_TO_YOUR_TEMPLATE.md')
```

Override `PATH_TO_YOUR_TEMPLATE.md` by the relative path to your custom template :)

Don't forget to use variables, in the template they should be referenced as `$VAR1`, and
in Python, you should pass a `dict`, with `{'var1': 'hello world'}`.

For instance, if you have a template called `hello_world.md`, with the following content:

```
Hello $NAME!
```

You should call this template as:

```py
vars = {'name': 'John Doe'}
report = Report(template='hello_world.md', variables=vars)
```

Now, you can generate a example report running:
```py
report.to_md('example_report.md')
```

The file `example_report.md` will have the following content:

```
Hello John Doe
```