from softioc import builder
import asyncio

class Epics:
    def __init__(self, cam_dat_eps, initial_ctr_param_values, init_dict=None):
        device_name, control_param_names, fit_param_names = self.load_names(init_dict)

        builder.SetDeviceName(device_name)

        ### Datanalyzer records
        #roi boundaries
        self.ao_roi_x_start = builder.aOut(control_param_names['roi_x_start'],
                                           initial_value=initial_ctr_param_values['roi_x_start'],
                                           on_update_name=lambda v, n: self.on_control_params_update('roi', 'roi_x_start', v),
                                           always_update=True)
        self.ao_roi_x_stop = builder.aOut(control_param_names['roi_x_stop'],
                                          initial_value=initial_ctr_param_values['roi_x_stop'],
                                          on_update_name=lambda v, n: self.on_control_params_update('roi', 'roi_x_stop', v),
                                          always_update=True)
        self.ao_roi_y_start = builder.aOut(control_param_names['roi_y_start'],
                                           initial_value=initial_ctr_param_values['roi_y_start'],
                                           on_update_name=lambda v, n: self.on_control_params_update('roi', 'roi_y_start', v),
                                           always_update=True)
        self.ao_roi_y_stop = builder.aOut(control_param_names['roi_y_stop'],
                                          initial_value=initial_ctr_param_values['roi_y_stop'],
                                          on_update_name=lambda v, n: self.on_control_params_update('roi', 'roi_y_stop', v),
                                          always_update=True)

       #fit_area
        self.ao_factor = builder.aOut(control_param_names['factor'],
                                      initial_value=float(initial_ctr_param_values['factor']),
                                      on_update_name=lambda v, n: self.on_control_params_update('fit_area', 'factor', v),
                                      always_update=True)
        self.ao_threshold = builder.aOut(control_param_names['threshold'],
                                         initial_value=initial_ctr_param_values['threshold'],
                                         on_update_name=lambda v, n: self.on_control_params_update('fit_area', 'threshold', v),
                                         always_update=True)
        self.ao_median_flt= builder.aOut(control_param_names['median_flt'],
                                         initial_value=initial_ctr_param_values['median_flt'],
                                         on_update_name=lambda v, n: self.on_control_params_update('fit_area', 'median_flt', v),
                                         always_update=True)

       #gauss_model
        self.ao_sampled = builder.aOut(control_param_names['sampled'],
                                       initial_value=initial_ctr_param_values['sampled'],
                                       on_update_name=lambda v, n: self.on_control_params_update('g_model', 'sampled', v),
                                       always_update=True)

       #fit parameter
        self.ai_amplitude = builder.aIn(fit_param_names['amplitude'], initial_value=0)
        self.ai_center_x = builder.aIn(fit_param_names['center_x'], initial_value=0)
        self.ai_center_y = builder.aIn(fit_param_names['center_y'], initial_value=0)
        self.ai_sigma_x = builder.aIn(fit_param_names['sigma_x'], initial_value=0)
        self.ai_sigma_y = builder.aIn(fit_param_names['sigma_y'], initial_value=0)
        self.ai_rotation = builder.aIn(fit_param_names['rotation'], initial_value=0)
        self.ai_offset = builder.aIn(fit_param_names['offset'], initial_value=0)


        # Boilerplate get the IOC started
        builder.LoadDatabase()
        self.cam_dat_eps = cam_dat_eps

        #settings saved for later
        self.device_name, self.control_param_names, self.fit_param_names = device_name, control_param_names, fit_param_names

    def load_names(self, init_dict=None):
        default_device_name = "CAMERA"
        default_control_param_names = {'roi_x_start': 'AO_ROI_X_START',
                                       'roi_x_stop': 'AO_ROI_X_STOP',
                                       'roi_y_start': 'AO_ROI_Y_START',
                                       'roi_y_stop': 'AO_ROI_Y_STOP',
                                       'factor': 'AO_FACTOR',
                                       'threshold': 'AO_THRESHOLD',
                                       'median_flt': 'AO_MEDIAN_FLT',
                                       'sampled': 'AO_SAMPLED'}
        default_fit_param_names = {'amplitude': 'AI_AMPLITUDE',
                                   'center_x': 'AI_CENTER_X',
                                   'center_y': 'AI_CENTER_Y',
                                   'sigma_x': 'AI_SIGMA_X',
                                   'sigma_y': 'AI_SIGMA_Y',
                                   'rotation': 'AI_ROTATION',
                                   'offset': 'AI_OFFSET'}

        device_name = default_device_name
        control_param_names = default_control_param_names
        fit_param_names = default_fit_param_names

        if init_dict:
            if 'device_name' in init_dict:
                device_name = init_dict['device_name']
            if 'control_params' in init_dict:
                for param, name in init_dict['control_params'].items():
                    control_param_names[param] = name
            if 'fit_params' in init_dict:
                for param, name in init_dict['fit_params'].items():
                    control_param_names[param] = name

        return device_name, control_param_names, fit_param_names


    def on_control_params_update(self, area, control_param_name, value):
        self.cam_dat_eps.on_control_params_update(area, control_param_name, value)
        print("in ", area, control_param_name, " change to ", value)

    def get_epics_settings(self):
        settings = {}
        settings['device_name'] = self.device_name
        settings['control_params'] = self.control_param_names
        settings['fit_params'] = self.fit_param_names
        return settings

    def set_fit_params(self, param_list):
        # order of params: [amplitude, centerx, centery, sigmax, sigmay, rot, offset]
        if param_list != []:
            self.ai_amplitude.set(param_list[0])
            self.ai_center_x.set(param_list[1])
            self.ai_center_y.set(param_list[2])
            self.ai_sigma_x.set(param_list[3])
            self.ai_sigma_y.set(param_list[4])
            self.ai_rotation.set(param_list[5])
            self.ai_offset.set(param_list[6])
        else:
            print("Empty param list")

    async def run(self):
        while True:
            params = self.cam_dat_eps.get_current_params()
            self.set_fit_params(params)
            print("new params set")
            await asyncio.sleep(2)

    def run_sync(self):
        params = self.cam_dat_eps.get_current_params()
        self.set_fit_params(params)
        print("new params set")


if __name__ == '__main__':

    init_dict_example = \
        {"device_name": "ALMUT",
         "control_params":
             {'roi_x_start': 'X_START',
              'roi_x_stop': 'X_STOP',
              'roi_y_start': 'Y_START',
              'roi_y_stop': 'Y_STOP',
              'factor': 'FACTOR',
              'threshold': 'THRESHOLD',
              'median_flt': 'MEDIAN_FLT',
              'sampled': 'SAMPLED'},
         "fit_params":
             {'amplitude': 'AMPLITUDE',
            'center_x': 'CENTER_X',
            'center_y': 'CENTER_Y',
            'sigma_x': 'SIGMA_X',
            'sigma_y': 'SIGMA_Y',
            'rotation': 'ROT',
            'offset': 'SET'}
         }


    initial_ctr_param_values = {'roi_x_start': 1,
                                       'roi_x_stop': 1,
                                       'roi_y_start': 1,
                                       'roi_y_stop': 1,
                                       'factor': 1,
                                       'threshold': 1,
                                       'median_flt': 1,
                                       'sampled': 1}

    #check if init okay
    epics = Epics(None, initial_ctr_param_values, init_dict_example)