from django.db import models


class Mesh(models.Model):

    filename = models.CharField()
    signature = models.CharField()
    category = models.CharField()

    model_type = models.IntegerField()
    model_style = models.IntegerField()
    face_count = models.IntegerField()

    generated = models.BooleanField()
    segent = models.BooleanField()

    def __str__(self):
        return self.filename

    class Meta:
        db_table = '"meshes"'

from mesh_server.mesh_model import Mesh
m = Mesh()
m.save()

