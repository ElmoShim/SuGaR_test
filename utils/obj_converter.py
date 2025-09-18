import vtk
import tqdm
import argparse
from pathlib import Path
from vtk.util import numpy_support


def convertTextureToPointAttribute(polyData, textureImageData):

    pointData = polyData.GetPointData()
    tcoords = pointData.GetTCoords()
    numOfPoints = pointData.GetNumberOfTuples()
    assert numOfPoints == tcoords.GetNumberOfTuples(), "Number of texture coordinates does not equal number of points"
    textureSamplingPointsUv = vtk.vtkPoints()
    textureSamplingPointsUv.SetNumberOfPoints(numOfPoints)
    for pointIndex in range(numOfPoints):
        uv = tcoords.GetTuple2(pointIndex)
        textureSamplingPointsUv.SetPoint(pointIndex, uv[0], uv[1], 0)

    textureSamplingPointDataUv = vtk.vtkPolyData()
    uvToXyz = vtk.vtkTransform()
    textureImageDataSpacingSpacing = textureImageData.GetSpacing()
    textureImageDataSpacingOrigin = textureImageData.GetOrigin()
    textureImageDataSpacingDimensions = textureImageData.GetDimensions()
    uvToXyz.Scale(textureImageDataSpacingDimensions[0] / textureImageDataSpacingSpacing[0],textureImageDataSpacingDimensions[1] / textureImageDataSpacingSpacing[1], 1)
    uvToXyz.Translate(textureImageDataSpacingOrigin)
    textureSamplingPointDataUv.SetPoints(textureSamplingPointsUv)
    transformPolyDataToXyz = vtk.vtkTransformPolyDataFilter()
    transformPolyDataToXyz.SetInputData(textureSamplingPointDataUv)
    transformPolyDataToXyz.SetTransform(uvToXyz)
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetInputConnection(transformPolyDataToXyz.GetOutputPort())
    probeFilter.SetSourceData(textureImageData)
    probeFilter.Update()


    rgbPoints = probeFilter.GetOutput().GetPointData().GetScalars()

    colorArray = vtk.vtkUnsignedCharArray()
    colorArray.SetName("RGB")
    colorArray.SetNumberOfComponents(3)
    colorArray.SetNumberOfTuples(numOfPoints)
    for pointIndex in range(numOfPoints):
        rgb = rgbPoints.GetTuple3(pointIndex)
        colorArray.SetTuple3(pointIndex, rgb[0], rgb[1], rgb[2])
    colorArray.Modified()

    polyData.GetPointData().SetScalars(colorArray)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default="input/A11/init_scaled.obj" )
    parser.add_argument("--output", type=Path, default=".")
    args = parser.parse_args()

    
    inputfile = args.input


    output_filename = inputfile.with_suffix(".ply").name
    output_filename_dec = output_filename[:-4] + "_decimated.ply"

    output_filename = args.output.joinpath(output_filename)
    output_filename_dec = args.output.joinpath(output_filename_dec)

    renWin = vtk.vtkRenderWindow()
    ren = vtk.vtkRenderer()
    ren.SetBackground(.5, .5, .5)
    renWin.AddRenderer(ren)

    # Import obj
    importer = vtk.vtkOBJImporter()
    importer.SetFileName(inputfile)
    importer.SetFileNameMTL(inputfile.with_suffix(".mtl"))
    importer.SetRenderWindow(renWin)
    importer.Update()

    # Get Actor

    actors = ren.GetActors()
    actors.InitTraversal()

    prop = vtk.vtkProperty()
    # prop.LightingOff()
    actors.ApplyProperties(prop)

    merger = vtk.vtkAppendPolyData()
    for idx, actor in enumerate( actors):
        polydata = actor.GetMapper().GetInput()
        texture = actor.GetTexture()
        image = texture.GetInput()

        image_size = image.GetDimensions()
        image_data = image.GetPointData().GetScalars()
        image_array = numpy_support.vtk_to_numpy(image_data)
        image_array = image_array[:,:3]
        image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(image_array))
        actor.GetProperty().SetAmbient(0)
        convertTextureToPointAttribute(polydata, image)
        merger.AddInputData(polydata)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(merger.GetOutputPort())
    cleaner.Update()

    result = cleaner.GetOutput()
    print("vertex # : ", result.GetNumberOfPoints())

    writer = vtk.vtkPLYWriter()
    writer.SetInputData(result)
    writer.SetColorModeToDefault()
    writer.SetArrayName("RGB")
    writer.SetFileName(output_filename)
    writer.Write()

    exit()

    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(result)
    decimate.SetTargetReduction(.9)
    decimate.VolumePreservationOn()
    decimate.AttributeErrorMetricOn()
    decimate.Update()

    decimated = decimate.GetOutput()
    print(decimated.GetNumberOfPoints())

    writer = vtk.vtkPLYWriter()
    writer.SetInputData(decimated)
    writer.SetColorModeToDefault()
    writer.SetArrayName("RGB")
    writer.SetFileName(output_filename_dec)
    writer.Update()

print("Done", output_filename_dec)

exit()
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(result)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.SetPosition(700, 0, 0)
ren.AddActor(actor)


iren = vtk.vtkRenderWindowInteractor()
iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
iren.SetRenderWindow(renWin)
renWin.Render()
iren.Start()