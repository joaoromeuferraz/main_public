from settings.portfolio import Settings as PortSettings


class Settings(PortSettings):
    start_date = '2014-01-02'
    end_date = '2020-09-21'
    rebal_frequency = 'M'
    rebal_time = 'CLOSE'
