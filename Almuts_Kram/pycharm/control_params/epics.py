from softioc import builder
import asyncio

class Epics:
    def __init__(self, cam_dat_eps):
        builder.SetDeviceName("ALMUT")

        #falls init dic, check ob alle werte abgedeckt sind, wenn ja setzte record names nicht mehr auf default

        records= {
            "roi_x_start":['AO_ROI_X_START', 0],
        }
        #print(record_names["roi_x_start"])


        ### Datanalyzer records
        #roi boundaries
        self.ao_roi_x_start = builder.aOut('AO_ROI_X_START', initial_value=1, always_update=True,
                               on_update_name=lambda v, n: self.on_update_data_a('roi', 'R', v))
        self.ao_roi_x_stop = builder.aOut('AO_ROI_X_STOP', initial_value=2, always_update=True,
                               on_update_name=lambda v, n: self.on_update_data_a('roi', 'roi_x_stop', v))
        self.ao_roi_y_start = builder.aOut('AO_ROI_Y_START', initial_value=3, always_update=True,
                               on_update_name=lambda v, n: self.on_update_data_a('roi', 'roi_y_start', v))
        self.ao_roi_y_stop = builder.aOut('AO_ROI_Y_STOP', initial_value=4, always_update=True,
                               on_update_name=lambda v, n: self.on_update_data_a('roi', 'roi_y_stop', v))

       #fit_area
        self.ao_factor = builder.aOut('AO_FACTOR', initial_value=5, always_update=True,
                               on_update_name=lambda v, n: self.on_update_data_a('fit_area', 'factor', v))
        self.ao_threshold = builder.aOut('AO_THRESHOLD', initial_value=5, always_update=True,
                               on_update_name=lambda v, n: self.on_update_data_a('fit_area', 'threshold', v))
        self.ao_median_flt= builder.aOut('AO_MEDIAN_FLT', initial_value=True, always_update=True,
                               on_update_name=lambda v, n: self.on_update_data_a('fit_area', 'median_flt', v))

       #gauss_model
        self.ao_sampled = builder.aOut('AO_sampled', initial_value=True, always_update=True,
                               on_update_name=lambda v, n: self.on_update_data_a('g_model', 'sampled', v))

       #fit parameter
        self.ai_amplitude = builder.aIn('AI_AMPLITUDE', initial_value=0)
        self.ai_center_x = builder.aIn('AI_CENTER_X', initial_value=0)
        self.ai_center_y = builder.aIn('AI_CENTER_Y', initial_value=0)
        self.ai_sigma_x = builder.aIn('AI_SIGMA_X', initial_value=0)
        self.ai_sigma_y = builder.aIn('AI_SIGMA_Y', initial_value=0)
        self.ai_rotation = builder.aIn('AI_ROTATION', initial_value=0)
        self.ai_offset = builder.aIn('AI_OFFSET', initial_value=0)


        # Boilerplate get the IOC started
        builder.LoadDatabase()
        self.cam_dat_eps = cam_dat_eps

    def on_update_data_a(self, area, ctr_param_name, value):
        print("in ", area, ctr_param_name, " change to ", value)


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
            params = self.cam_dat_eps.data_a.params
            self.set_fit_params(params)
            print("new params set")
            await asyncio.sleep(2)


#ao Controlwerte ankommen