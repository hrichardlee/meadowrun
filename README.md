# ![Meadowrun](meadowrun-logo-full.svg)

[![Join the chat at https://gitter.im/meadowdata/meadowrun](https://badges.gitter.im/meadowdata/meadowrun.svg)](https://gitter.im/meadowdata/meadowrun?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Meadowrun automates the tedious details of running your python code on AWS. Meadowrun
will
- choose the optimal set of EC2 on-demand or spot instances and turn them off when
  they're no longer needed
- deploy your code and libraries to the EC2 instances, so you don't have to worry about
  creating packages and building Docker images
- scale from a single function to thousands of parallel tasks

For more information see Meadowrun's [documentation](https://docs.meadowrun.io), the
[project homepage](https://meadowrun.io), or [join the chat on
Gitter](https://gitter.im/meadowdata/meadowrun)

## When would I use Meadowrun?

- You want to run distributed computations in python in AWS with as little fuss as
  possible. You don't want to build docker images as part of your iteration cycle, and
  you want to be able to take advantage of spot pricing without having to manage a
  cluster.
- You can't/don't want to run tests or analysis on your laptop and you want a better
  experience than SSHing into an EC2 machine.

## Quickstart

First, install Meadowrun using pip: 

```
pip install meadowrun
```

conda:

```
conda install -c defaults -c conda-forge -c meadowdata meadowrun
```

or poetry:

```
poetry add meadowrun
```

Next, assuming you've [configured the AWS
CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html)
and are a root/administrator user, you can run:

```python
import meadowrun
import asyncio

def run_meadowrun_function():
    return await meadowrun.run_function(
        lambda: sum(range(1000)) / 1000,
        meadowrun.AllocCloudInstance(
            logical_cpu_required=4,
            memory_gb_required=32,
            interruption_probability_threshold=15,
            cloud_provider="EC2"
        ),
        await meadowrun.Deployment.mirror_local()
    )

print(asyncio.run(run_meadowrun_function()))
```
