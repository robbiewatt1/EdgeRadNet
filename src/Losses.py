import torch

"""
The script includes the loss function to train the predictor network
"""


class EmitLoss(torch.nn.Module):
    """
    Loss function for the emittance prediction.
    """

    def __init__(self, r_mats_375, k_factor=1.0297, quad_l=0.1068,
                 device="cuda:0"):
        super(EmitLoss, self).__init__()
        self.r_mats_375 = r_mats_375
        self.quad_l = quad_l
        self.k_factor = k_factor
        self.device = device

    def get_single_emittance(self, beam_params):
        """
        Calculate the emittance from the beam parameters
        :param beam_params: Beam parameters
        :return: Emittance
        """
        emit_y2 = (beam_params[..., 0] * beam_params[..., 2]
                   - beam_params[..., 1]**2.)
        return emit_y2

    def ground_truth_scan_batch(self, size_2_batch, field_batch):
        """
        Calculate the beam parameters at the quad and emittance.
        :param size_2: beam size at PR11375
        :param field: field strength of the quad
        :return: beam parameters at the quad and emittance
        """

        beam_params_x = []
        beam_params_y = []
        emit_x2 = []
        emit_y2 = []
        for i, (size_2, field) in enumerate(zip(size_2_batch, field_batch)):
            quad_x, quad_y = self.get_quad_matrix(field)
            r_mat_x = self.r_mats_375[0] @ quad_x
            coeff_matrix_x = torch.stack([
                r_mat_x[:, 0, 0] ** 2.,
                2. * r_mat_x[:, 0, 0] * r_mat_x[:, 0, 1],
                r_mat_x[:, 0, 1] ** 2.]).T
            beam_params_x.append(torch.linalg.lstsq(
                coeff_matrix_x, size_2[:, 0]).solution)

            r_mat_y = self.r_mats_375[1] @ quad_y
            coeff_matrix_y = torch.stack([
                r_mat_y[:, 0, 0] ** 2.,
                2. * r_mat_y[:, 0, 0] * r_mat_y[:, 0, 1],
                r_mat_y[:, 0, 1] ** 2.]).T
            beam_params_y.append(torch.linalg.lstsq(
                coeff_matrix_y, size_2[:, 1]).solution)

            emit_x2.append(beam_params_x[-1][0] * beam_params_x[-1][2]
                           - beam_params_x[-1][1] ** 2.)
            emit_y2.append(beam_params_y[-1][0] * beam_params_y[-1][2]
                           - beam_params_y[-1][1] ** 2.)

        beam_params_x = torch.stack(beam_params_x)
        beam_params_y = torch.stack(beam_params_y)
        emit_x2 = torch.stack(emit_x2)
        emit_y2 = torch.stack(emit_y2)
        return beam_params_x, beam_params_y, emit_x2, emit_y2

    def propogae_forward(self, field, beam_params):
        """
        Propogate the beam parameters through the quad
        """
        _, quad_y = self.get_quad_matrix(field)

        r_mat_y = self.r_mats_375[1] @ quad_y
        beam_params_matrix = torch.stack(
            [beam_params[:, 0], beam_params[:, 1],
             beam_params[:, 1], beam_params[:, 2]], dim=1).reshape((-1, 2, 2))
        beam_params_matrix = r_mat_y @ beam_params_matrix @ r_mat_y.transpose(1, 2)
        return torch.stack([beam_params_matrix[:, 0, 0],
                            beam_params_matrix[:, 0, 1],
                            beam_params_matrix[:, 1, 1]], dim=1)

    def get_quad_matrix(self, field):
        """
        Get the quad matrix for the given field strength.
        :param field: field strength in Tesla
        :return: quad matrix for x and y planes
        """
        field = self.k_factor * field
        field = field.cfloat()
        # Add small part to field to avoid division by zero
        field = field + 1e-8
        quad_mat_x = torch.stack([
            torch.cos(field**0.5 * self.quad_l),
            field**-0.5 * torch.sin(field**0.5 * self.quad_l),
            -field**0.5 * torch.sin(field**0.5 * self.quad_l),
            torch.cos(field**0.5 * self.quad_l)])
        quad_mat_y = torch.stack([
            torch.cosh(field**0.5 * self.quad_l),
            field**-0.5 * torch.sinh(field**0.5 * self.quad_l),
            field**0.5 * torch.sinh(field**0.5 * self.quad_l),
            torch.cosh(field**0.5 * self.quad_l)])
        quad_mat_x = quad_mat_x.permute(1, 0).reshape(-1, 2, 2).real
        quad_mat_y = quad_mat_y.permute(1, 0).reshape(-1, 2, 2).real
        return quad_mat_x, quad_mat_y
