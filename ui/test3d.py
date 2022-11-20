import sys
from PySide6.QtCore import (Property, QObject, QPropertyAnimation, Signal)
from PySide6.QtGui import (QGuiApplication, QMatrix4x4, QQuaternion, QVector3D)
from PySide6.Qt3DCore import (Qt3DCore)
from PySide6.Qt3DExtras import (Qt3DExtras)

import random

from PySide6.QtWidgets import QApplication


class OrbitTransformController(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self._target = None
        self._matrix = QMatrix4x4()
        self._radius = 1
        self._angle = 0

    def setTarget(self, t):
        self._target = t

    def getTarget(self):
        return self._target

    def setRadius(self, radius):
        if self._radius != radius:
            self._radius = radius
            self.updateMatrix()
            self.radiusChanged.emit()

    def getRadius(self):
        return self._radius

    def setAngle(self, angle):
        if self._angle != angle:
            self._angle = angle
            self.updateMatrix()
            self.angleChanged.emit()

    def getAngle(self):
        return self._angle

    def updateMatrix(self):
        self._matrix.setToIdentity()
        self._matrix.rotate(self._angle, QVector3D(0, 1, 0))
        self._matrix.translate(self._radius, 0, 0)
        if self._target is not None:
            self._target.setMatrix(self._matrix)

    angleChanged = Signal()
    radiusChanged = Signal()
    angle = Property(float, getAngle, setAngle, notify=angleChanged)
    radius = Property(float, getRadius, setRadius, notify=radiusChanged)


class Window(Qt3DExtras.Qt3DWindow):
    def __init__(self):
        super().__init__()

        # Camera
        self.camera().lens().setPerspectiveProjection(45, 16 / 9, 0.1, 1000)
        self.camera().setPosition(QVector3D(0, 0, 60))
        self.camera().setViewCenter(QVector3D(0, 0, 0))

        # For camera controls
        self.createScene()
        self.camController = Qt3DExtras.QOrbitCameraController(self.rootEntity)
        self.camController.setLinearSpeed(50)
        self.camController.setLookSpeed(180)
        self.camController.setCamera(self.camera())

        self.setRootEntity(self.rootEntity)

    def createScene(self):
        # Root entity
        self.rootEntity = Qt3DCore.QEntity()

        # Material
        self.material = Qt3DExtras.QPhongMaterial(self.rootEntity)


        # Torus
        # self.torusEntity = Qt3DCore.QEntity(self.rootEntity)
        # self.torusMesh = Qt3DExtras.QTorusMesh()
        # self.torusMesh.setRadius(5)
        # self.torusMesh.setMinorRadius(1)
        # self.torusMesh.setRings(100)
        # self.torusMesh.setSlices(20)
        #
        # self.torusTransform = Qt3DCore.QTransform()
        # self.torusTransform.setScale3D(QVector3D(1.5, 1, 0.5))
        # self.torusTransform.setRotation(QQuaternion.fromAxisAndAngle(QVector3D(1, 0, 0), 45))
        #
        # self.torusEntity.addComponent(self.torusMesh)
        # self.torusEntity.addComponent(self.torusTransform)
        # self.torusEntity.addComponent(self.material)
        i = 1
        sphereEntity = Qt3DCore.QEntity(self.rootEntity)
        sphereEntity.setEnabled(True)
        sphereEntity.setParent(self.rootEntity)
        sphereMaterial = Qt3DExtras.QPhongMaterial(self.rootEntity)
        sphereMesh = Qt3DExtras.QSphereMesh()
        sphereMesh.setRadius(2)
        sphereTransform = Qt3DCore.QTransform()
        sphereTransform.setTranslation(QVector3D(0, 0.5 * i, 0.5 * i))
        sphereEntity.addComponent(sphereMesh)
        sphereEntity.addComponent(sphereMaterial)
        sphereEntity.addComponent(sphereTransform)

        # Sphere
        for i in range(0):
            radius = (random.random()*2)+1
            rotationtime = random.randrange(8000, 20000)
            orbitradius = random.randrange(15,45)
            orbitradius = 20
            sphereEntity = Qt3DCore.QEntity()
            sphereEntity.setParent(self.rootEntity)
            sphereMaterial = Qt3DExtras.QPhongMaterial(self.rootEntity)
            sphereMaterial.setParent(sphereEntity)
            sphereMesh = Qt3DExtras.QSphereMesh()
            sphereMesh.setParent(sphereEntity)

            sphereMesh.setRadius(radius)

            sphereTransform = Qt3DCore.QTransform()
            sphereTransform.setTranslation(QVector3D(0, 0.5*i,0.5*i))
            sphereTransform.setParent(sphereEntity)


            # controller = OrbitTransformController(sphereTransform)
            # controller.setTarget(sphereTransform)
            # controller.setRadius(20)

            # sphereRotateTransformAnimation = QPropertyAnimation(sphereTransform)
            # sphereRotateTransformAnimation.setTargetObject(controller)
            # sphereRotateTransformAnimation.setPropertyName(b"angle")
            # sphereRotateTransformAnimation.setStartValue(0)
            # sphereRotateTransformAnimation.setEndValue(360)
            # sphereRotateTransformAnimation.setDuration(rotationtime)
            # sphereRotateTransformAnimation.setLoopCount(-1)
            # sphereRotateTransformAnimation.start()

            # sphereEntity.addComponent(sphereMesh)
            # sphereEntity.addComponent(sphereTransform)
            sphereEntity.addComponent(self.material)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = Window()
    view.show()
    view.show()
    sys.exit(app.exec())