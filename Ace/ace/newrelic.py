import newrelic.agent


class CustomRequestAttrs:
    REQUEST = "request"
    PATH = "path"
    STATUS = "status"
    METHOD = "method"
    SERVICE = "service"
    DURATION = "duration"


def init_thread():
    """
    This function is called when a new executor thread is created, it's purpose is to register the thread with New Relic

    See:
     - https://forum.newrelic.com/s/hubtopic/aAX8W0000008b3lWAA/python-multithreading
     - https://forum.newrelic.com/s/hubtopic/aAX8W0000008Zm6WAE/problem-with-python-multiprocessing
     - https://forum.newrelic.com/s/hubtopic/aAX8W0000008ZnAWAU/relic-solution-python-agent-and-the-multiprocessing-library
    """
    import newrelic.agent

    newrelic.agent.register_application()


def add_transaction_attr(name: str, value: str | int | float | bytes, service_name: str | None = None):
    """Custom attributes to New Relic's transaction"""
    add_transaction_attrs(((name, value),), service_name)


def add_transaction_attrs(attrs: tuple[tuple[str, str | int | float | bytes], ...], service_name: str | None = None):
    """Custom attributes to New Relic's transaction"""
    if service_name:
        newrelic.agent.add_custom_attributes([(f"ace.{service_name}.{name}", value) for name, value in attrs])
    else:
        newrelic.agent.add_custom_attributes([(f"ace.{name}", value) for name, value in attrs])
